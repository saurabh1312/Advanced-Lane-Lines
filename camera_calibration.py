import numpy as np
import cv2
import os
import glob
import pickle

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
object_points = []  # 3d points in real world space
image_points = []  # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for file_name in images:
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # If found, add object points, image points
    if ret:
        object_points.append(objp)
        image_points.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        output_filename = 'output_images/camera_calibration/' + os.path.basename(file_name)
        cv2.imwrite(output_filename, img)

# Open and image for reference
img = cv2.imread('./camera_cal/calibration2.jpg')
img_size = (img.shape[1], img.shape[0])

# Calibrate camera using object and image points
_, mtx, dist, _, _ = cv2.calibrateCamera(object_points, image_points, img_size, None, None)

# Save calibration output for later
dist_pickle = {'mtx': mtx, 'dist': dist}
pickle.dump(dist_pickle, open('./calibration_pickle.p', 'wb'))



