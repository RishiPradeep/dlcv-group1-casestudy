import os
import sys
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import mrcnn.config
import mrcnn.utils
import mrcnn.model as modellib
from mrcnn import visualize

# Load model configuration and weights
class InferenceConfig(mrcnn.config.Config):
    NAME = "coco_inference"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir='./', config=config)
model.load_weights('mask_rcnn_coco.h5', by_name=True)

# Load and preprocess image
image_path = 'C:/Users/rishi/OneDrive/Desktop/Stuff/Projects/my-yolov8-app/public/db41_jpg.rf.a8cc022f9efec9c0d9a383de9379f184.jpg'  # Replace with your image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform detection
results = model.detect([image], verbose=1)
r = results[0]

# Display the results
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            r['scores'], r['class_ids'], class_names=model.class_names)
