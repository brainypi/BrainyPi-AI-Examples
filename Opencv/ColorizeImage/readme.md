## Image Colorization on BrainyPi 
## Description
We will be implementing image colorization application on BrainyPi which converts the black and white image into color image.

## Install Opencv on BrainyPi
```sh
pip3 install opencv-python
```

## Clone the repository
  ```sh
  git clone https://github.com/brainypi0/BrainyPi-AI-Examples.git
  cd BrainyPi-AI-Examples/Opencv/ColorizeImage
  ```
## Download the model
- Download from [drive](https://drive.google.com/file/d/1bJuUPBc8zRjj-VX5vCtffQ3ldOWvaQi0/view?usp=sharing)

## Run Image colorization example
```sh
python3 colorize.py 
```
- Input
  - Put all images you want to colorize in test_samples folder
  - Use Opencv VideoCapture to convert black and white video to color video
- Output
  - Shows the Label with probability on terminal and the detected objects image.
    
## Sample Input and Output images
<img src="test_samples/car.jpg" alt="drawing" width="1000"/>
<img src="results/result.jpg" />
  
## Original Documentation and Model
https://medium.com/towards-data-science/colorizing-old-b-w-photos-and-videos-with-the-help-of-ai-76ba086f15ec
