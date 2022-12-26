## People Counter With Pytorch on BrainyPi
## Description
We will be implementing People Counter application on BrainyPi which is trained on coco dataset including person class.

## Install Pytorch on BrainyPi
```sh
pip3 install torch torchvision
```
## Clone the repository
  ```sh
  git clone https://github.com/brainypi/BrainyPi-AI-Examples.git
  cd BrainyPi-AI-Examples/Pytorch/people-counter
  ```
## Run Image classfication example
```sh
python people_counter.py mb1-ssd models/mobilenet-v1-ssd-mp-0_675.pth models/voc-model-labels.txt video.mp4
```
- Input
  - Parameter1: Model name
  - Parameter2: Model path
  - Parameter3: Labels file path
  - Parameter4: Input Video `In case this parameter is not provided, it will take the input from camera directly.`


- Output
  - Shows the Label with probability on terminal




## Original Document
https://pytorch.org/tutorials/intermediate/realtime_rpi.html
