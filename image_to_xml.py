import os
import time
import tensorflow as tf
import cv2
import numpy as np

from object_detection.utils import label_map_util
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from IPython.display import HTML
from base64 import b64encode
import lxml.builder
import lxml.etree

#Path to saved model  

PATH_TO_SAVED_MODEL = "D:\\wobot\\training_demo\\exported-models\\newmodel5\\saved_model"

# Load label map and obtain class names and ids
#label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
category_index=label_map_util.create_category_index_from_labelmap("D:\\wobot\\training_demo\\annotations\\label_map.pbtxt",use_display_name=True)


def visualise_on_image(image, bboxes, labels, scores, thresh):
    (h, w, d) = image.shape
    for bbox, label, score in zip(bboxes, labels, scores):
        if score > thresh:
            xmin, ymin = int(bbox[1]*w), int(bbox[0]*h)
            xmax, ymax = int(bbox[3]*w), int(bbox[2]*h)
            cx = int((xmin+xmax)/2)
            cy = int((ymin+ymax)/2)
            pixel = image[cy, cx]
            b, g, r = int(pixel[0]), int(pixel[1]), int(pixel[2])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (b,g,r), 2)
            cv2.putText(image, f"{label}: {int(score*100)} %", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (b,g,r), 2)
    return image



def toPascalVoc(image_path,filename, shapes):

    img = cv2.imread(image_path)
    dimensions = img.shape

    maker = lxml.builder.ElementMaker()
    xml = maker.annotation(
        maker.folder(),
        maker.filename(image_path),
        maker.database(),  
        maker.annotation(),  
        maker.image(),  
        maker.size(
            maker.height(str(dimensions[0])),
            maker.width(str(dimensions[1])),
            maker.depth(str(dimensions[2])),
        ),
        maker.segmented(),
    )

 
    for shape in shapes:
        (xmin, ymin), (xmax, ymax) = shape["points"]
        xmin, xmax = sorted([xmin, xmax])
        ymin, ymax = sorted([ymin, ymax])

        xml.append(
            maker.object(
                maker.name(shape["label"]),
                maker.pose(),
                maker.truncated(),
                maker.difficult(),
                maker.bndbox(
                    maker.xmin(str(xmin)),
                    maker.ymin(str(ymin)),
                    maker.xmax(str(xmax)),
                    maker.ymax(str(ymax)),
                ),
            )
        )
    output_xml = "D:\\wobot\\training_demo\\stage3\\output_xml"
    with open(os.path.join(output_xml,filename) + ".xml", "wb") as f:
        f.write(lxml.etree.tostring(xml, pretty_print=True))




if __name__ == '__main__':
    
    # Load the model
    print("Loading saved model ...")
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    print("Model Loaded!")
    
    input_folder = "D:\\wobot\\training_demo\\stage3\\test_images"
    output_folder = "D:\\wobot\\training_demo\\stage3\\detected_images"
    
    for filename in os.listdir(input_folder):
      img = cv2.imread(os.path.join(input_folder,filename))
      image_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]

      detections = detect_fn(input_tensor)

      score_thresh = 0.15   # Minimum threshold for object detection
      max_detections = 5

      scores = detections['detection_scores'][0, :max_detections].numpy()
      bboxes = detections['detection_boxes'][0, :max_detections].numpy()
      labels = detections['detection_classes'][0, :max_detections].numpy().astype(np.int64)
      labels = [category_index[n]['name'] for n in labels]

      detected_image =visualise_on_image(img, bboxes, labels, scores, score_thresh)

      cv2.putText(detected_image, f"detcetd image", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
      cv2.imwrite(os.path.join(output_folder,filename), img)
      
      (h, w, d) = img.shape
    #   xmin, ymin = int(bboxes[0][1]*w), int(bboxes[0][0]*h)
    #   xmax, ymax = int(bboxes[0][3]*w), int(bboxes[0][2]*h) 
      shapes =[] 
      for bbox, label, score in zip(bboxes, labels,scores):
            xmin, ymin = int(bbox[1]*w), int(bbox[0]*h)
            xmax, ymax = int(bbox[3]*w), int(bbox[2]*h)
            if (score<score_thresh):
                shapes.append({"label": label, "points": [(xmin,ymin),(xmax,ymax)]})

      toPascalVoc(os.path.join(input_folder,filename),filename, shapes)


