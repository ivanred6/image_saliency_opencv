# Packages Import
import argparse
import cv2
import os
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to BING objectness saliency model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-n", "--max-detections", type=int, default=10,
	help="maximum # of detections to examine")
args = vars(ap.parse_args())
ap.add_argument("-d", "--diff", required=True, help="Differentiator")
ap.add_argument("-a", "--here", required=False, help="Ignores SaveTo and uses CWD to save output")

# Load our input image
image = cv2.imread(args["image"])

di = args["diff"]