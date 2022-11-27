## Object Detection With TFLite on BrainyPi 
## Description
We will be implementing object detection application on BrainyPi using [starter model](https://www.tensorflow.org/lite/examples/object_detection/overview) trained on COCO 2017 dataset.

## Install TFlite on BrainyPi
```sh
pip3 install tflite-runtime
```

## Clone the repository
  ```sh
  git clone https://github.com/brainypi0/BrainyPi-AI-Examples.git
  cd BrainyPi-AI-Examples/TFLite/ObjectDetection
  ```
## Install prerequisites
```sh
bash install-prerequisites.sh
```

## Run Object Detection example
```sh
python3 objectDetection.py 
```

- Input
  - Parameter 1: '--model', help='Provide the path to the TFLite file, default is models/model.tflite'
                    (default='models/model.tflite')
  - Parameter 2: '--labels', help='Provide the path to the Labels, default is models/labels.txt'
                    (default='models/labels.txt')
  - Parameter 3: '--image_dir', help='Name of the single image to perform detection on'
                    (default='images/test1.jpg')
  - Parameter 4: '--threshold', help='Minimum confidence threshold for displaying detected objects'
                    (default=0.5)
  - Parameter 5: '--save_dir', help='Directory path to save the image',
                    default='results/result.jpg')
- Output
  - Shows predicted object bounding boxes with confidence scores.
  
## Original Documentation and Model
https://www.tensorflow.org/lite/examples/object_detection/overview

https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/default/1

## References
https://github.com/armaanpriyadarshan/TensorFlow-2-Lite-Object-Detection-on-the-Raspberry-Pi
