## Image Recognition With ONNX on BrainyPi 
## Description
We will be implementing image recognition application on ONNX which is trained on imagenet which has 1000 classes. List is here

## Install Onnx Runtime on BrainyPi
```sh
pip3 install onnx-runtime
```
## Clone the repository
  ```sh
  git clone https://github.com/brainypi0/BrainyPi-AI-Examples.git
  cd BrainyPi-AI-Examples/Onnx/ImageClassification
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
https://onnxruntime.ai/docs/get-started/with-python.html
