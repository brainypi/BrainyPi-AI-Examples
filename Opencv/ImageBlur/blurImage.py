# importing libraries
import cv2
import numpy as np
import argparse

FLAGS=None

parser = argparse.ArgumentParser()
parser.add_argument(
    '--image_file',
    type=str,
    default='peacock.jpg',
    help='Absolute path to image file.'
)
FLAGS, unparsed = parser.parse_known_args()


# Read image with opencv  
image = cv2.imread(FLAGS.image_file)
  
# Write original image
cv2.imwrite('OriginalImage.jpg', image)
  
# Gaussian Blur
Gaussian = cv2.GaussianBlur(image, (7, 7), 0)
cv2.imwrite('GaussianBlurring.jpg', Gaussian)
  
# Median Blur
median = cv2.medianBlur(image, 5)
cv2.imwrite('MedianBlurring.jpg', median)  
  
# Bilateral Blur
bilateral = cv2.bilateralFilter(image, 9, 75, 75)
cv2.imwrite('BilateralBlurring.jpg', bilateral)
cv2.destroyAllWindows()

