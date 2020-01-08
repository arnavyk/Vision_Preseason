import cv2
import os
import numpy as np
import cv2
import glob

# Add function to do resolution 640 by 360
vidcap = cv2.VideoCapture(1)
success,image = vidcap.read()
count = 0
success = True
os.system("rm -rf images")
os.system("mkdir images")
while success:
    if(count%3==0):
        success,image = vidcap.read()
        print('Read a new frame #%d: ' % count, success)
        cv2.imwrite("images/frame%d.jpg" % count, image)   
        cv2.imshow("Video", image)
          # save frame as JPEG file
    if cv2.waitKey(10) == 27:                     # exit if Escape is hit
        break
    count += 1
