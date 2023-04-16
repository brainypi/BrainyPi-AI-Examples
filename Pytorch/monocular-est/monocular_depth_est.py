#pip install transformers

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

checkpoint = "vinvino02/glpn-nyu"

image_processor = AutoImageProcessor.from_pretrained(checkpoint)
model = AutoModelForDepthEstimation.from_pretrained(checkpoint)

from PIL import Image
import requests
import cv2

"""
## TO DOWNLOAD FROM NET

url = "https://unsplash.com/photos/HwBAsSbPBDU/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MzR8fGNhciUyMGluJTIwdGhlJTIwc3RyZWV0fGVufDB8MHx8fDE2Nzg5MDEwODg&force=true&w=640"
image = Image.open(requests.get(url, stream=True).raw)
image # to see the input
"""
image=Image.open('input.jpg') #to load from local system


pixel_values = image_processor(image, return_tensors="pt").pixel_values
import torch

with torch.no_grad():
    outputs = model(pixel_values)
    predicted_depth = outputs.predicted_depth

import numpy as np

# interpolate to original size
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
).squeeze()
output = prediction.numpy()

"""
# to visualize output

formatted = (output * 255 / np.max(output)).astype("uint8")
depth = Image.fromarray(formatted)
depth 
"""


# visualize output
import matplotlib.pyplot as plt
plt.imsave('output.jpg',output)

#plt.imsave('in.jpg',image) to save input image if downloaded from internet

"""
img=Image.open('/content/output.jpg') #to visualize output
img.show()
"""
