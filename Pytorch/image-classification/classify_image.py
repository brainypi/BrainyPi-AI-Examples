import time
import argparse
import torch
import numpy as np
from torchvision import models, transforms

import cv2
from PIL import Image

torch.backends.quantized.engine = 'qnnpack'
FLAGS = None
#cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
#cap.set(cv2.CAP_PROP_FPS, 36)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Read labels file
with open("labels.txt", "r") as fp:
    lines = fp.readlines()

net = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
# jit model to take it from ~20fps to ~30fps
net = torch.jit.script(net)

started = time.time()
last_logged = time.time()
#frame_count = 0

def classify(imagePath, num_pred):
    started = time.time()
    last_logged = time.time()
    frame_count = 0
    with torch.no_grad():
        while True:
            # read frame
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (224,224))
            #if not ret:
            #    raise RuntimeError("failed to read frame")

            # convert opencv output from BGR to RGB
            image = image[:, :, [2, 1, 0]]
            permuted = image

            # preprocess
            input_tensor = preprocess(image)

            # create a mini-batch as expected by the model
            input_batch = input_tensor.unsqueeze(0)

            # run model
            output = net(input_batch)
            # do something with output ...

            # log model performance
            frame_count += 1
            now = time.time()
            if now - last_logged > 1:
                print(f"{frame_count / (now-last_logged)} fps")
                last_logged = now
                frame_count = 0
            break

    top = list(enumerate(output[0].softmax(dim=0)))
    top.sort(key=lambda x: x[1], reverse=True)
    for idx, val in top[:num_pred]:
        print(f"{val.item()*100:.2f}% {lines[idx]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--image_file',
      type=str,
      default='peacock.jpg',
      help='Absolute path to image file.'
    )
    parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=5,
      help='Display this many predictions.'
    )
    num_pred = 5
    FLAGS, unparsed = parser.parse_known_args()
    image = FLAGS.image_file
    num_pred = FLAGS.num_top_predictions 
    classify(image, num_pred)
