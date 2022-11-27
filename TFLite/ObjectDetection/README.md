## Object Detection With TFLite on BrainyPi 
## Description
We will be implementing object detection application on BrainyPi [using starter model](https://www.tensorflow.org/lite/examples/object_detection/overview) trained on COCO 2017 dataset.

## Install TFlite on BrainyPi
```sh
pip3 install tflite-runtime
```

## Clone the repository
  ```sh
  git clone https://github.com/brainypi0/BrainyPi-AI-Examples.git
  cd BrainyPi-AI-Examples/TFLite/Object Detection
  ```
## Install prerequisites
```sh
./install-prerequisites.sh
```

## Run Object Detection example
```sh
python3 objectDetection.py 
```

- Input
  - Parameter1: --image_dir: Image file location. (default='images/car.jpg')
  - Parameter2: --save_dir: Directory path to save the result image. (default='results/result.jpg')
- Output
  - Shows the segmentation and overlay with class labels in the output image.
  
## Original Documentation and Model
https://www.tensorflow.org/lite/examples/object_detection/overview
https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/default/1

## References
armaanpriyadarshan/TensorFlow-2-Lite-Object-Detection-on-the-Raspberry-Pi: This repository contains everything you need to run TensorFlow Lite on the Raspberry Pi with the updated TensorFlow 2 version.
