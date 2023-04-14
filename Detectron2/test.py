import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import pickle
import matplotlib.pyplot as plt
import requests
import cv2

'''
to download an input image
response = requests.get('http://images.cocodataset.org/val2017/000000439715.jpg')
open("input.jpg", "wb").write(response.content)
'''
import argparse

parser = argparse.ArgumentParser(description='Run object detection on an input image')
parser.add_argument('input_file', type=str, help='path to input image file')

args = parser.parse_args()

input_file_path = args.input_file

im = cv2.imread(input_file_path)
#im = cv2.imread("Carspotters.jpg")
#fig, ax = plt.subplots(figsize=(18, 8))
#ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

cfg = get_cfg()

#cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) #to dwnld config file
with open('config.pkl', 'rb') as f:
    cfg = pickle.load(f)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") #to dwnld chk points
#cfg.MODEL.WEIGHTS ='model_weights.pth' to load downloaded check points

# If you don't have a GPU and CUDA enabled, the next line is required
cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)
outputs = predictor(im)

torch.save(predictor.model.state_dict(), "model_weights.pth")

with open('config.pkl', 'wb') as f:
    pickle.dump(cfg, f)

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#fig, ax = plt.subplots(figsize=(18, 8))
#ax.imshow(out.get_image()[:, :, ::-1])
output_image = out.get_image()[:, :, ::-1]

# Convert the output image from BGR to RGB color space
output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

# Save the output image as a PNG file
cv2.imwrite("output2.png", output_image)

