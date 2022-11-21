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

#saved model path

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

if __name__ == '__main__':
    
    # Load the model
    print("Loading saved model ...")
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    print("Model Loaded!")
    
    # Video Capture (video_file)
    video_capture = cv2.VideoCapture("D:\\wobot\\training_demo\\input.mp4")
    start_time = time.time()
    
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    #fps = int(video_capture.get(5))
    size = (frame_width, frame_height)
    
    #Initialize video writer
    fourcc = 0x00000021
    result = cv2.VideoWriter('result_video.mp4', fourcc ,15, size)

    while True:
      ret, frame = video_capture.read()
      if not ret:
          print('Unable to read video / Video ended')
          break
    
      frame = cv2.flip(frame, 1)
      image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
      # The model expects a batch of images, so also add an axis with `tf.newaxis`.
      input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]

      # Pass frame through detector
      detections = detect_fn(input_tensor)

      # Set detection parameters

      score_thresh = 0.3   # Minimum threshold for object detection
      max_detections = 2

      # All outputs are batches tensors.
      # Convert to numpy arrays, and take index [0] to remove the batch dimension.
      # We're only interested in the first num_detections.
      scores = detections['detection_scores'][0, :max_detections].numpy()
      bboxes = detections['detection_boxes'][0, :max_detections].numpy()
      labels = detections['detection_classes'][0, :max_detections].numpy().astype(np.int64)
      labels = [category_index[n]['name'] for n in labels]

      # Display detections
      visualise_on_image(frame, bboxes, labels, scores, score_thresh)

      end_time = time.time()
      fps = int(1/(end_time - start_time))
      start_time = end_time
      cv2.putText(frame, f"FPS: {fps}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
      #cv2_imshow(frame)
      
      #Write output video
      result.write(frame)

    video_capture.release()
