import os
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
print(tf.__version__)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', help='Name of the single image to perform detection on',
                    default='images/low.jpg')
parser.add_argument('--save_dir', help='Directory to save the results',
                    default='results/result.jpg')
                    
args = parser.parse_args() 
img = args.image_dir
opt = args.save_dir

interpreter = tf.lite.Interpreter(model_path = 'model.tflite')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input Shape:", input_details[0]['shape'])
print("Input Type:", input_details[0]['dtype'])
print("Output Shape:", output_details[0]['shape'])
print("Output Type:", output_details[0]['dtype'])

IMG_HEIGHT=600
IMG_WIDTH=400

def preprocess_image(image_path):
    original_image = Image.open(image_path)
    width, height = original_image.size
    preprocessed_image = original_image.resize(
        (
            IMG_HEIGHT,
            IMG_WIDTH
        ),
        Image.ANTIALIAS)
    preprocessed_image = tf.keras.preprocessing.image.img_to_array(preprocessed_image)
    preprocessed_image = preprocessed_image.astype('float32') / 255.0
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    
    return original_image, preprocessed_image

def infer_tflite(image):
    interpreter = tf.lite.Interpreter(model_path = 'model.tflite')
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_index, image)
    interpreter.invoke()
    raw_prediction = interpreter.tensor(output_index)
    output_image = raw_prediction()

    output_image = output_image.squeeze() * 255.0
    output_image = output_image.clip(0, 255)
    output_image = output_image.reshape(
        (np.shape(output_image)[0], np.shape(output_image)[1], 3)
    )
    output_image = Image.fromarray(np.uint8(output_image))
    return output_image

def save_results(images, titles, figure_size=(12, 12)):
    fig = plt.figure(figsize=figure_size)
    plt.imshow(images[1])
    fig.savefig(opt)
    




original_image, preprocessed_image = preprocess_image(img)
output_image = infer_tflite(preprocessed_image)
save_results(
    [original_image, output_image],
    ["Original Image", "Enhanced Image"],
    (20, 12),
)

