import os
import time
#import tensorflow.compat.v2 as tf
import tensorflow as tf
from pathlib import Path
import glob
#import tkinter
from pycocotools import mask as maskUtils
from PIL import Image
from skimage import measure
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, mapping
import cv2

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

INPUT_DIR='/home/chintu/Desktop/maskrcnn/maskrcnn_wth_tensorflow_object_detection_api/training_data/test'
IMAGE_DIR=os.path.join(INPUT_DIR, "images")
LABEL_FILE='/home/chintu/Desktop/maskrcnn/maskrcnn_wth_tensorflow_object_detection_api/training_data/labelmap.pbtxt'
MODEL_DIR='/home/chintu/Desktop/maskrcnn/maskrcnn_wth_tensorflow_object_detection_api/training_data/saved_model/frozen_inference_graph.pb'
OUTPUT_DIR='/home/chintu/Desktop/inference_output'

image_files= glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))

NUM_CLASSES=1
IMAGE_SIZE = (12, 12)

# load model 
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(MODEL_DIR, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


#load labels
label_map = label_map_util.load_labelmap(LABEL_FILE)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)



def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def remove_by_probability(in_dict,probability=0.01):
  output={}
  output['num_detections']=0
  output['detection_boxes']=[]
  output['detection_masks']=[]
  output['detection_classes']=[]
  output['detection_scores']=[]
  for i in range(in_dict['num_detections']):
    if in_dict['detection_scores'][i]>probability:
      output['detection_boxes'].append(in_dict['detection_boxes'][i])
      output['detection_masks'].append(in_dict['detection_masks'][i])
      output['detection_classes'].append(in_dict['detection_classes'][i])
      output['detection_scores'].append(in_dict['detection_scores'][i])
  output['detection_boxes']=np.array(output['detection_boxes'])
  output['detection_masks']=np.array(output['detection_masks'])
  output['detection_classes']=np.array(output['detection_classes'])
  output['detection_scores']=np.array(output['detection_scores'])
  print(output['detection_scores'].shape[0])
  output['num_detections']=output['detection_scores'].shape[0]
  return output


_final_object2={}
for image_path in image_files:
  image = Image.open(image_path)
  filename= os.path.basename(image_path)
  print(filename)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  print(image_np.shape)

  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  
  image_np_expanded = np.expand_dims(image_np, axis=0)
  print(image_np_expanded.shape)
  
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  output_dict = remove_by_probability(output_dict,probability=0.1)

  print(output_dict['num_detections'])
  #print(output_dict['detection_boxes'])
  #print(output_dict['detection_classes'].shape)
  #print(output_dict['detection_scores'].shape)
  #print(output_dict['detection_masks'].shape)


  
"""
fp = open(os.path.join(INPUT_DIR,"machine_annotation.json"), "w")
import json
print("Writing JSON...")
fp.write(json.dumps(_final_object2))
fp.close()
"""