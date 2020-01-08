import numpy as np
import cv2
import glob
import json

def calc_chessboard_corners(images_path, example_image_path):
    '''Calculates the chessboard corners and points using the given images'''
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(images_path)

    gray = cv2.imread(example_image_path, cv2.COLOR_BGR2GRAY)

    count = -1
    for fname in images:
        count = count + 1
        if(count%6 == 0):
            print(count)
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (6,9),None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9, 6), corners2,ret)
                cv2.imshow('img',img)
                cv2.waitKey(1)

    cv2.destroyAllWindows()
    return (objpoints, imgpoints, gray)


def create_img(objpoints, imgpoints, gray, mtx, dist, chessboard_image_path):
    '''Creates a chessboard based on the given parameters'''
    img = cv2.imread(chessboard_image_path)
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    return (img, newcameramtx, w, h, roi)

def crop_and_undistort_image(mtx, dist, img, new_camera_mtx, width, height, roi):
    '''Crops and undistorts the given image with the given values'''

    # undistort
    mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,new_camera_mtx,(width, height),5)
    dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibresult.png',dst)

def calc_mean_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    '''Calculates the re projection error based on the given parameters'''
    mean_error = 0.0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    mean_error = mean_error/len(objpoints)

    return mean_error

def calibrate_camera_with_chessboard(images_path, example_image_path, chessboard_image_path):
    '''Calibrates the connected camera with a checkerboard pattern'''
    objpoints, imgpoints, gray = calc_chessboard_corners(images_path, example_image_path)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    img, new_camera_mtx, width, height, roi = create_img(objpoints, imgpoints, gray, mtx, dist, chessboard_image_path)
    crop_and_undistort_image(mtx, dist, img, new_camera_mtx, width, height, roi)
    mean_error = calc_mean_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)

    return (mtx, dist, mean_error)


def writeToJSON(images_paths, example_image_path, chessboard_image_path, camera_names, data_file_name, data_file_type):
    '''Takes values for a camera matrix, distortion coefficients, and the mean error, and writes them to a json file'''
    data = {}
    data["Cameras"] = {}
    for first in range(len(camera_names)):
        mtx, dist, mean_error = calibrate_camera_with_chessboard(images_paths[first], example_image_path, chessboard_image_path)
        data["Cameras"][camera_names[first]]= {}
        data["Cameras"][camera_names[first]]['Calibration Values'] = {}
        data["Cameras"][camera_names[first]]['Error'] = mean_error
        data["Cameras"][camera_names[first]]['Distortion'] = []

        calibration_meanings = ["Focal Length X", "Zero", "Optical Canter X", "Zero", "Focal Length Y", "Optical Center Y", "Zero", "Zero", "One"]
        meaning_index = 0
        for index in range(len(mtx)):
            data["Cameras"][camera_names[first]]['Calibration Values']['Matrix %s' % (index + 1)] = []
            for second in range(len(mtx[index])):
                data["Cameras"][camera_names[first]]['Calibration Values']['Matrix %s' % (index + 1)].append({calibration_meanings[meaning_index]: mtx[index][second]})
                meaning_index += 1

        distortion_meanings = ["k1", "k2", "p1", "p2", "p3"]
        meaning_index = 0
        for index in range(len(dist[0])):
            data["Cameras"][camera_names[first]]['Distortion'].append({distortion_meanings[meaning_index]: dist.tolist()[0][index]})
            meaning_index += 1

        with open(data_file_name, data_file_type) as outfile:
            json.dump(data, outfile, indent=4)

def readFromJSON(data_file_name):
    with open(data_file_name) as json_file:
        data =  json.load(json_file)
        return data

def compareCameras(filename):
    data = readFromJSON(filename)
    camera1 = data["Cameras"]["Camera 1"]
    camera2= data["Cameras"]["Camera 2"]
    if(camera1["Calibration Values"] != camera2["Calibration Values"]):
        return False
    # elif(camera1["Error"] != camera2["Error"]):
    #     return False
    # elif(camera1["Distortion"] != camera2["Distortion"]):
    #     return False
    else:
        return True

if __name__ == "__main__":
    images_paths = ['/Users/arnav/git/Robotics/cameracalibration/camera_one_images/*.jpg', '/Users/arnav/git/Robotics/cameracalibration/camera_two_images/*.jpg']
    example_image_path = "/Users/arnav/git/Robotics/cameracalibration/hi.jpg"
    chessboard_image_path = '/Users/arnav/git/Robotics/cameracalibration/left12.jpg'
    camera_names = ["Camera 1", "Camera 2"]
    data_file_name = 'data.json'
    data_file_type = 'w'

    mtx, dist, mean_error = calibrate_camera_with_chessboard(images_paths[0], example_image_path, chessboard_image_path)
    writeToJSON(images_paths, example_image_path, chessboard_image_path, camera_names, data_file_name, data_file_type)
    data = readFromJSON(data_file_name)

    print(compareCameras(data_file_name))
    print("Camera Matrix: ", mtx)
    print ("total error: ", mean_error)
    print("Distortion: ", dist)