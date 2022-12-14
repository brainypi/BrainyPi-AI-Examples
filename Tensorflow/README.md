## Image Recognition With TensorFlow on RaspberryPi / BrainyPi
## Description
We will be implementing image recognition application on BrainyPi which is trained on imagenet which has 1000 classes. List is here

## Install Tensorflow on BrainyPi
```sh
pip3 install tensorflow
```
- Installs version = 2.10.0

## Clone the repository
  ```sh
  git clone https://github.com/brainypi/BrainyPi-AI-Examples.git
  cd BrainyPi-AI-Examples/Tensorflow
  ```

## Run Image classfication example
```sh
python3 classify_image.py --image_file peacock.jpg
```
- Input
  - Parameter1: --image_file: Image file location
  - Parameter2: ----num_top_predictions: No of predictions you want to see
- Output
  - Shows the Label with probability on terminal
## Original Document
https://www.instructables.com/Image-Recognition-With-TensorFlow-on-Raspberry-Pi/
