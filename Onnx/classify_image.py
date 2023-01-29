import argparse
import numpy as np
import onnxruntime as rt

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

FLAGS = None
parser = argparse.ArgumentParser()
parser.add_argument(
    '--image_file',
    type=str,
    default='peacock.jpg',
    help='Absolute path to image file.'
)
parser.add_argument(
    '--num_top_predictions',
    type=int,
    default=5,
    help='Display this many predictions.'
)

FLAGS, unparsed = parser.parse_known_args()

# Accept imagename from command line
img_path = FLAGS.image_file

output_names=['predictions']

img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Model path
output_path = "resnet50.onnx"
providers = ['CPUExecutionProvider']
m = rt.InferenceSession(output_path, providers=providers)
onnx_pred = m.run(output_names, {"input": x})

# Print the results
print('ONNX Predicted:', decode_predictions(onnx_pred[0], top=FLAGS.num_top_predictions)[0])
