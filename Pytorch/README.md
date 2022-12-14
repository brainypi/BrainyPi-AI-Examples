## Image Recognition With Pytorch on BrainyPi
## Description
We will be implementing image recognition application on BrainyPi which is trained on imagenet which has 1000 classes. List is here

## Install Pytorch on BrainyPi
```sh
pip3 install torch torchvision
```
## Clone the repository
  ```sh
  git clone https://github.com/brainypi/BrainyPi-AI-Examples.git
  cd BrainyPi-AI-Examples/Pytorch
  ```
## Run Image classfication example
```sh
python3 classify_image.py --image_file
```
- Input
  - Parameter1: --image_file: Image file location
  - Parameter2: ----num_top_predictions: No of predictions you want to see
- Output
  - Shows the Label with probability on terminal
## Original Document
https://pytorch.org/tutorials/intermediate/realtime_rpi.html
