import numpy
import cv2
import json

#Callback for trackbars - does nothing because checking the trackbar position here creates synchronization issues so it's done in the while loop instead
def nothing(x):
    pass

#Takes the min and max HSV values and writes it to a file with specified file name
#File is in JSON and file_name must include .json
def writeToJSON(file_name):
    data = {}
    data.append({"min_h": min_h})
    data.append({"min_s": min_s})
    data.append({"min_v": min_v})
    data.append({"max_h": max_h})
    data.append({"max_s": max_s})
    data.append({"max_v": max_v})

    #Opens a new file and writes (denoted by the "w") the json to this file
    with open(file_name, "w") as outfile:
        json.dump(data, outfile, indent = 4)

#Initializes window with tuners
cv2.namedWindow("Tuner")
cv2.createTrackbar('Min Hue','Tuner',0,255,nothing)
cv2.createTrackbar('Max Hue','Tuner',0,255,nothing)
cv2.createTrackbar('Min Saturation','Tuner',0,255,nothing)
cv2.createTrackbar('Max Saturation','Tuner',0,255,nothing)
cv2.createTrackbar('Min Value','Tuner',0,255,nothing)
cv2.createTrackbar('Max Value','Tuner',0,255,nothing)
min_h = 0
min_s = 0
min_v = 0
max_h = 255
max_s = 255
max_v = 255
frame_count = 0
width, height = 600, 400

#Sets up video stream to be used in while loop
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

display = cap.read()[1]

while True:
    frame = cap.read()[1]
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, (min_h, min_s, min_v), (max_h, max_s, max_v))
    cv2.imshow("Tuner", display)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    min_h = cv2.getTrackbarPos("Min Hue", "Tuner")
    max_h = cv2.getTrackbarPos("Max Hue", "Tuner")
    min_s = cv2.getTrackbarPos("Min Saturation", "Tuner")
    max_s = cv2.getTrackbarPos("Max Saturation", "Tuner")
    min_v = cv2.getTrackbarPos("Min Value", "Tuner")
    max_v = cv2.getTrackbarPos("Max Value", "Tuner")
    #low_value = (min_h, min_s, min_v)
    #high_value = (max_h, max_s, max_v)
    
    #This combines the mask with the original image
    #Set display to mask or imshow mask above if you just want to see the mask
    display = cv2.bitwise_and(frame, frame, mask=mask)
writeToJSON("color.json")
cv2.destroyAllWindows()
