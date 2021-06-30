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
import pickle
import geopandas as gpd
import rasterio.windows
from rasterio.windows  import bounds
import fiona

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

INPUT_DIR='/home/chintu/Desktop/maskrcnn/maskrcnn_wth_tensorflow_object_detection_api/training_data/val/'
IMAGE_DIR=os.path.join(INPUT_DIR, "images")
#PICKLE_DIR= os.path.join(INPUT_DIR, "chip_dfs.pkl")
#CRS_DIR = os.path.join(INPUT_DIR, "crs.pkl")
LABEL_FILE='/home/chintu/Desktop/maskrcnn/maskrcnn_wth_tensorflow_object_detection_api/training_data/labelmap.pbtxt'
MODEL_DIR='/home/chintu/Desktop/maskrcnn/maskrcnn_wth_tensorflow_object_detection_api/training_data/saved_model_05062021/frozen_inference_graph.pb'
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

def to_actualcoordinates(poly, reference_bounds, ncol, nrows=1024,ncols=1024):
    minx, miny, maxx, maxy = reference_bounds
    w_poly, h_poly = (maxx - minx, maxy - miny)

    x_scaler =  w_poly / ncols
    y_scaler = -1*h_poly / nrows
    p=[]
    for x, y in poly.exterior.coords:
        p.append((x*x_scaler + minx, maxy + y*y_scaler))
    main_poly=Polygon(p)
    return main_poly

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
  output['num_detections']=output['detection_scores'].shape[0]
  return output

def remove_overlapping_contours(r,input_image):
  resized_height, resized_width = int(input_image.shape[0]*0.269) , int(input_image.shape[1]* 0.269)
  _mask = np.zeros((resized_height,resized_width), dtype=np.uint8)
  for i in range(len(r['detection_boxes'])):
    if r['detection_scores'][i] < 0.01:
      r['detection_scores'][i] = 0.0
      continue
    (startY, startX, endY, endX) = r['detection_boxes'][i].astype("int")
    _mask_tmp = r['detection_masks'][i]
    contours, hierarchy = cv2.findContours(np.asarray(_mask_tmp, dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blank_image = np.zeros((input_image.shape[0], input_image.shape[1]))
    if not contours:
      continue
    for _index in range(len(contours[0])):
      contours[0][_index][0][0] += startX
      contours[0][_index][0][1] += startY
    cv2.drawContours(blank_image, contours, -1, (255, 255, 255), -1)
    _msk =  cv2.resize(np.asarray(blank_image, dtype=np.uint8), (resized_width,resized_height), interpolation=cv2.INTER_CUBIC)
    inter = np.bitwise_or(_mask, _msk)
    inter_area = np.count_nonzero(_mask) + np.count_nonzero(_msk) - np.count_nonzero(inter)
    if inter_area > 0:
      if (inter_area / np.count_nonzero(_msk)) < 0.5:
        _mask = np.bitwise_or(inter, _mask)
      else:
        r['detection_scores'][i] = 0.0
    else:
      _mask = np.bitwise_or(inter, _mask)

  return r

all_polygon=[]
poly_bounds={}
'''
chip_dfs=pickle.load(open(PICKLE_DIR,"rb"))
crs = pickle.load(open(CRS_DIR,"rb"))
for key in chip_dfs.keys():
  l=chip_dfs.get(key)
  poly_bounds[key]=l['chip_poly'].bounds
'''

_final_object2={}
for image_path in image_files:
  image = Image.open(image_path)
  filename= os.path.basename(image_path)
  print(filename)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  print(output_dict['num_detections'])
  output_dict = remove_overlapping_contours(output_dict,image_np)
  print(output_dict['num_detections'])
  #output_dict = remove_by_probability(output_dict)
  #print(output_dict['num_detections'])


  # Visualization of the results of a detection.
  """
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      min_score_thresh = 0.1,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      max_boxes_to_draw=output_dict['num_detections'])
  
  Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
  cv2.imwrite(os.path.join(Path(OUTPUT_DIR),filename),image_np)
  fig = plt.figure(figsize=IMAGE_SIZE)
  plt.axis('off')
  matplotlib.use('TkAgg')
  plt.imshow(image_np)
  #plt.show()
  plt.savefig(os.path.join(Path(OUTPUT_DIR),filename))
  plt.close()
  """
  
  #for creating annotation  file
  image_object={}
  image_object["filename"]=filename
  file_attributes={}
  file_attributes["width"]=image.size[0]
  file_attributes["height"]=image.size[1]
  image_object["file_attributes"]=file_attributes
  regions=[]
  for _idx,class_id in enumerate(output_dict['detection_classes']):
    if class_id==1 and _idx<output_dict['detection_masks'].shape[0]:
      #mask = output_dict['detection_masks'].astype(np.uint8)[:, :, _idx]
      mask = output_dict['detection_masks'][_idx]
      bbox = np.around(output_dict['detection_boxes'][_idx], 1)
      _mask = maskUtils.encode(np.asfortranarray(mask))
      _mask["counts"] = _mask["counts"].decode("UTF-8")
      contours = measure.find_contours(mask, 0.5)
      #print(contours)
      polygon_points=[]
      for contour in contours:
        contour = np.flip(contour, axis=1)
        all_points_x=contour[:,0].tolist()
        all_points_y=contour[:,1].tolist()
        for i in range(contour.shape[0]):
          polygon_points.append((contour[i,0] ,contour[i,1]))
        polygon=Polygon(polygon_points)
        #all_polygon.append(to_actualcoordinates(polygon, poly_bounds[filename.replace("val","train").replace(".jpg","")], image.size[0], image.size[1]))
        region={}
        poly2={}
        poly2["name"]="polygon"
        poly2["all_points_x"]=all_points_x
        poly2["all_points_y"]=all_points_y
        region["shape_attributes"]=poly2
        #region_attributes={}
        #region_attributes["bbox"]=[bbox[1],  bbox[0], bbox[3]-bbox[1],bbox[2]-bbox[0]]
        #region["region_attributes"]=region_attributes
        regions.append(region)
  image_object["regions"]=regions
  _final_object2[filename.replace("val","train").replace(".jpg","")]=image_object
  
'''
outpath=Path(os.path.join(INPUT_DIR,"shape"))
Path(outpath).mkdir(parents=True, exist_ok=True)
output_shape=outpath / 'new_shape.shp'

multi_polygon=MultiPolygon(all_polygon)

#multi_polygon.to_file(output_shape, driver='ESRI Shapefile', encoding='utf-8')
schema= {
    'geometry': 'Polygon',
    'properties' : {'id':'int'},
}

with fiona.open(output_shape, 'w', crs=crs['crs'], driver='ESRI Shapefile', encoding='utf-8', schema=schema) as c:
    for _idx,poly in enumerate(all_polygon):
        c.write({'geometry': mapping(poly),
                 'properties': {'id': _idx},
        })
'''
fp = open(os.path.join(INPUT_DIR,"machine_annotation.json"), "w")
import json
print("Writing JSON...")
fp.write(json.dumps(_final_object2))
fp.close()
