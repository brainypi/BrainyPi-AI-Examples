import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

model = torch.hub.load('ultralytics/yolov5', 'custom', path='/content/yolov5/runs/train/exp3/weights/last.pt', force_reload=True)

results = model("/image4.png")
results.print()
%matplotlib inline 
plt.imshow(np.squeeze(results.render()))
plt.show()
plt.imsave("results.png", np.squeeze(results.render())