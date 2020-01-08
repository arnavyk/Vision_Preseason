import cv2
import numpy as np

# Read Image
cap = cv2.VideoCapture(0)
_, frame = cap.read()
size = frame.shape


# Extracts image using color detection
img = cv2.imread('/Users/andraliu/Desktop/blue.jpg')  

# convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define range of green color in HSV
lower_green = np.array([45, 50, 50])
upper_green = np.array([80, 255, 255])

# get only green colors from image
# mask is the black and white output
mask = cv2.inRange(hsv, lower_green, upper_green)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(img, img, mask= mask)


# box is a bounding rectangle
box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
box = np.int0(box)

#2D image points - camera image
#Use color filter and draw the rectangle around vision target to find the points and distances in pixels
image_points = np.array([
                            (xul, yul),     # Vision target upper left
                            (xdl, yul),     # Vision target down left
                            (xur, yur),     # Vision target upper right
                            (xdr, ydr),     # Vision target down right
                        ], dtype="double")
                        
# 3D model points - real world object
model_points = np.array([
                            (0.0, 0.0, 0.0),          # Vision target upper left
                            (0.0, 0.0, -431.8),       # Vision target lower left
                            (996.95, 0.0, 0.0),       # Vision target upper right
                            (996.95, 0.0, -431.8),    # Vision target lower right
])
                        
# Calibration stats/Camera internals
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                         [[focal_length,0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
)

                         
dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)
# Insert code to find the actual distance and print it

    # draw a bounding box around the image and display it
    box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
    box = np.int0(box)
    cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
    cv2.putText(image, "%.2fft" % (inches / 12),
        (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
        2.0, (0, 255, 0), 3)
    cv2.imshow("image", image)
    cv2.waitKey(0)

 # Display image
cv2.imshow("Output", frame)
cv2.waitKey(1)

def find_largest_contour(image, debug=False):
    '''
    Finds the largest contour in the inputted image.
    Returns the minimum area rectangle of that contour.
    If no contours are found, returns -1.
    '''
    # Blurs the image for better contour detection accuracy
    #blur_image = cv2.medianBlur(image, 5)
    #blur_image = cv2.GaussianBlur(blur_image, (5, 5), 0)
    blur_image = image.copy()

    # Finds ALL the contours of the image
    # Note: the tree and chain things could probably be optimized better.
    contours = cv2.findContours(blur_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[1]
    if len(contours) != 0:
        # Find the biggest area contour
        biggest_contour = max(contours, key=cv2.contourArea)
        # Creating a rotated minimum area rectangle
        rect = cv2.minAreaRect(biggest_contour)
        if debug:
            cv2.imshow("Blurred", blur_image)
            draw_image = image.copy()
            cv2.drawContours(draw_image, contours, -1, (255, 0, 0), 2)
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)
            cv2.drawContours(draw_image, [box_points], 0, (0, 255, 0), 2)
            cv2.imshow("Contours / Rectangle", image)
        return rect
    else:
        return -17

