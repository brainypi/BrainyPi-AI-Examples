# BrainyPi-AI-Examples
This Document will give you a step wise instuctions for installing AI frameworks on BrainyPi and using it.

## Installing Frameworks for AI 
## Tensorflow
### 1. Installing Tensorflow
- Run this command on terminal
  ```sh
  pip3 install tensorflow
  ```
  - Installs version: 2.10.0
### 2. TF sample program to classify 1000 imagenet classes
- Clone the repository and navigate to folder
  ```sh
  git clone https://github.com/brainypi/BrainyPi-AI-Examples.git
  cd BrainyPi-AI-Examples/Tensorflow
  ```
- Run Image classfication example. classifies Imagenet 1000 classes.
  ```
  python3 classify_image.py --image_file peacock.jpg
  ```
  - Input
    - Parameter1(Default: peacock.jpg): --image_file: Image file location
    - Parameter2(Default: 5): ----num_top_predictions: No of predictions you want to see
  - Output
    - Shows the Classified class label with probability on terminal
   
## TFlite
### 1. Installing Tflite 

- Run this command on terminal
  ```sh
  pip3 install tflite-runtime
  ```
### 2. TFlite sample program to classify 1000 imagenet classes
- Clone the repository and navigate to folder
  ```sh
  git clone https://github.com/brainypi/BrainyPi-AI-Examples.git
  cd BrainyPi-AI-Examples/TFLite
  ```
- Run Image classfication example. classifies Imagenet 1000 classes.
  ```
  python3 classify_image.py --image_file peacock.jpg
  ```
  - Input
    - Parameter1(Default: peacock.jpg): --image_file: Image file location
    - Parameter2(Default: 5): ----num_top_predictions: No of predictions you want to see
  - Output
    - Shows the Classified class label with probability on terminal

## Pytorch
### 1. Installing Pytorch
- Run this command on terminal
  ```sh
  pip3 install torch torchvision
  ```
### 2. Pytorch Sample program to classify 1000 imagenet classes
- Clone the repository and navigate to folder
  ```sh
  git clone https://github.com/brainypi/BrainyPi-AI-Examples.git
  cd BrainyPi-AI-Examples/Pytorch
  ```
- Run Image classfication example. classifies Imagenet 1000 classes.
  ```
  python3 classify_image.py --image_file peacock.jpg
  ```
  - Input
    - Parameter1(Default: peacock.jpg): --image_file: Image file location
    - Parameter2(Default: 5): ----num_top_predictions: No of predictions you want to see
  - Output
    - Shows the Classified class label with probability on terminal
    
## Opencv
### 3. Installing Opencv
- Run this command on terminal
  ```sh
  pip3 install opencv-python
  ```
 
### 2. Opencv Sample program to blur the image
- Clone the repository and navigate to folder
  ```sh
  git clone https://github.com/brainypi/BrainyPi-AI-Examples.git
  cd BrainyPi-AI-Examples/Opencv
  ```
- Run Image Blur example.
  ```
  python3 blurImage.py --image_file peacock.jpg
  ```
  - Input
    - Parameter1(Default: peacock.jpg): --image_file: Image file location
  - Output
    - Stores output images with respective blue method names.
    
## More Examples and Sample Projects on AI 
   - [TFLite Example](https://github.com/brainypi/BrainyPi-AI-Examples/tree/main/TFLite)
