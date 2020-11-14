import argparse
import cv2

# We first build the argument parser, then parse args...
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Path to input image")
args = vars(parser.parse_args())

# Load our input image
img = cv2.imread(args["image"])

# Initialise OpenCV's static saliency SPECTRAL RESIDUAL DETECTOR and compute saliency map
saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
(success, saliencyMap) = saliency.computeSaliency(img)
saliencyMap = (saliencyMap * 255).astype("uint8")
cv2.imshow("Image", img)
cv2.imshow("Output", saliencyMap)
cv2.waitKey(0)
test = input("hello")