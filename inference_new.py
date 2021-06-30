#prediction and result 
import os
import sys
import time
import numpy as np
import skimage.io
import pickle
import geopandas as gpd
import rasterio.windows
from rasterio.windows  import bounds 
import shapely
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, mapping
import fiona
from pathlib import Path
# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3. 
# A quick one liner to install the library 
# !pip install git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils 
from skimage import measure

import coco #a slightly modified version

from mrcnn.evaluate import build_coco_results, evaluate_coco
from mrcnn.dataset import ForafricDataset
from mrcnn import visualize


import zipfile
import urllib.request
import shutil
import glob
import tqdm
import random
#------------------------------------------prepare my data location-------------------------------------------------------------------#
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils


PRETRAINED_MODEL_PATH = os.path.join(ROOT_DIR,"data1024_3","trained_model_1024.h5")
LOGS_DIRECTORY = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

#put images in test folder before testing the model
IMAGE_DIR = os.path.join(ROOT_DIR, "data1024_3", "test", "images")
PICKLE_DIR= os.path.join(ROOT_DIR, "data1024_3", "chip_dfs.pkl")
CRS_DIR = os.path.join(ROOT_DIR, "data1024_3", "crs.pkl")
#------------------------------------------------------------------------------------------------------------------------------#
#i intentiate my  configuration
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    STEPS_PER_EPOCHS = 500
    VALIDATION_STEP = 50
    NUM_CLASSES = 1+1
    IMAGE_MAX_DIM=1024
    IMAGE_MIN_DIM=1024
    NAME = "ForAfricPro"
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3
    TRAIN_ROIS_PER_IMAGE = 200
    DETECTION_MAX_INSTANCES = 100
    DETECTION_GT_INSTANCES = 100

config = InferenceConfig()
config.display()
#--------------------------------------------------------------------------------------------------------------------------------------#
#I initiate my  model 
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

model_path = PRETRAINED_MODEL_PATH

# or if you want to use the latest trained model, you can use : 
#model_path = model.find_last()[1]

model.load_weights(model_path, by_name=True)
#----------------------------------------------------------------------------------------------------------------------------------------#
#i run prediction for single image 
class_names = ['BG', 'field'] # In our case, we have 1 class
#file_names = next(os.walk(IMAGE_DIR))[2]
#print(file_names)
#random_image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
#random_image = skimage.io.imread('/home/shreekanthajith/intello_satellite/Agri/data/train-og/images/COCO_train2016_000000100000.jpg')

#predictions = model.detect([random_image]*config.BATCH_SIZE, verbose=1) # We are replicating the same image to fill up the batch_size
#p = predictions[0]
#visualize.display_instances(random_image,p['rois'], p['masks'], p['class_ids'], 
#                            class_names, p['scores'])

#the follow code is to evaluate the model and compute the average precision 

# Gather all JPG files in the test set as small batches

def to_actualcoordinates(poly, reference_bounds, ncol, nrows=config.IMAGE_SHAPE[0],ncols=config.IMAGE_SHAPE[1]):
    minx, miny, maxx, maxy = reference_bounds
    w_poly, h_poly = (maxx - minx, maxy - miny)

    x_scaler =  w_poly / ncols
    y_scaler = -1*h_poly / nrows
    p=[]
    for x, y in poly.exterior.coords:
        p.append((x*x_scaler + minx, maxy + y*y_scaler))
    main_poly=Polygon(p)
    return main_poly



chip_dfs=pickle.load(open(PICKLE_DIR,"rb"))
crs = pickle.load(open(CRS_DIR,"rb"))

#print(chip_dfs['COCO_train2016_100000100000']['chip_poly'].bounds)

all_polygon=[]
poly_bounds={}
for key in chip_dfs.keys():
    l=chip_dfs.get(key)
    poly_bounds[key]=l['chip_poly'].bounds

files = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
ALL_FILES=[]
_buffer = []
for _idx, _file in enumerate(files):
    if len(_buffer) == config.IMAGES_PER_GPU * config.GPU_COUNT:
        ALL_FILES.append(_buffer)
        _buffer = []
    _buffer.append(_file)

if len(_buffer) > 0:
    ALL_FILES.append(_buffer)

# Iterate over all the batches and predict
_final_object2={}
for files in tqdm.tqdm(ALL_FILES):
    images = [skimage.io.imread(x) for x in files]
    predoctions = model.detect(images, verbose=0)
    
    for _idx, r in enumerate(predoctions):
        image_object={}
        _file = files[_idx]
        image_id = int(_file.split("_")[-1].replace(".jpg",""))
        file_name=os.path.basename(_file)
        image_object["filename"]=file_name
        file_name=file_name.replace("val","train").replace(".jpg","")
        file_attributes={}
        file_attributes["width"]=images[_idx].shape[0]
        file_attributes["height"]=images[_idx].shape[1]
        image_object["file_attributes"]=file_attributes
        regions=[]
        bounds_data=poly_bounds[file_name]
        for _idx, class_id in enumerate(r["class_ids"]):
            if class_id == 1:
                mask = r["masks"].astype(np.uint8)[:, :, _idx]
                bbox = np.around(r["rois"][_idx], 1)
                bbox = [float(x) for x in bbox]
                _mask = maskUtils.encode(np.asfortranarray(mask))
                _mask["counts"] = _mask["counts"].decode("UTF-8")
                contours = measure.find_contours(mask, 0.5)
                polygon_points=[]
                for contour in contours:
                    contour = np.flip(contour, axis=1)
                    all_points_x=contour[:,0].tolist()
                    all_points_y=contour[:,1].tolist()
                    for i in range(contour.shape[0]):
                        polygon_points.append((contour[i,0] ,contour[i,1]))
                polygon=Polygon(polygon_points)
                all_polygon.append(to_actualcoordinates(polygon, poly_bounds[file_name], config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]))
                region={}
                poly2={}
                poly2["name"]="polygon"
                poly2["all_points_x"]=all_points_x
                poly2["all_points_y"]=all_points_y
                region["shape_attributes"]=poly2
                region_attributes={}
                region_attributes["bbox"]=[bbox[1],  bbox[0], bbox[3]-bbox[1],bbox[2]-bbox[0]]
                region["region_attributes"]=region_attributes
                regions.append(region)
        image_object["regions"]=regions
        _final_object2[file_name]=image_object

outpath=Path(r'shape/')
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



fp = open("annotation.json", "w")
import json
print("Writing JSON...")
fp.write(json.dumps(_final_object2))
fp.close()

