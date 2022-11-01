## Image Recognition With TensorFlow on RaspberryPi / BrainyPi
## Description
We will be implementing image recognition application on BrainyPi which is trained on imagenet which has 1000 classes. List is here

## Install Tensorflow on BrainyPi
```sh
pip3 install tensorflow
```
- Installs version = 2.10.0

## Run Image classfication example
```sh
python3 classify_image.py --image_file peacock.jpg
```
- Parameters
  - Parameter1: --image_file: Image file location
  - Parameter2: ----num_top_predictions: No of predictions you want to see

## Original Document
https://www.instructables.com/Image-Recognition-With-TensorFlow-on-Raspberry-Pi/
