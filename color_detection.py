# import the necessary packages
import argparse

import numpy as np

import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])

    # numbers go bgr in the arrays
    # blue
	# ([64, 0, 0], [255, 250, 250]) 
    # green
    # ([0, 64, 0], [250, 255, 250])
    # red
    # ([0, 0, 64], [250, 250, 255]),

# create NumPy arrays from the boundaries
lower = np.array([64, 0, 0], dtype = "uint8")
upper = np.array([255, 250, 250], dtype = "uint8")
# find the colors within the specified boundaries and apply the mask
mask = cv2.inRange(image, lower, upper)
output = cv2.bitwise_and(image, image, mask = mask)
cv2.imwrite("onlyblue.jpg", output)

lower = np.array([0, 64, 0], dtype = "uint8")
upper = np.array([250, 255, 250], dtype = "uint8")
mask = cv2.inRange(image, lower, upper)
output = cv2.bitwise_and(image, image, mask = mask)
cv2.imwrite("onlygreen.jpg", output)

lower = np.array([0, 0, 64], dtype = "uint8")
upper = np.array([250, 250, 255], dtype = "uint8")
mask = cv2.inRange(image, lower, upper)
output = cv2.bitwise_and(image, image, mask = mask)
cv2.imwrite("onlyred.jpg", output)





