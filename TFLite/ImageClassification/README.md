## Image Recognition With TFLite on BrainyPi 
## Description
We will be implementing image recognition application on BrainyPi which is trained on imagenet which has 1000 classes. List is here

## Install TFlite on BrainyPi
```sh
pip3 install tflite-runtime
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
https://blog.paperspace.com/tensorflow-lite-raspberry-pi/
