#!/usr/bin/env python3
# import the necessary packages
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import argparse
import imutils
import cv2
import sys
import json
import os

'''
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
'''
# load the image and perform pyramid mean shift filtering
# to aid the thresholding step

initial_image_path = os.path.expanduser('~/calibtration_images/initial_image.jpg')
rel_pos0_path = os.path.expanduser('~/calibtration_images/rel_pos0.jpg')
rel_pos1_path = os.path.expanduser('~/calibtration_images/rel_pos1.jpg')
rel_pos2_path = os.path.expanduser('~/calibtration_images/rel_pos2.jpg')
rel_pos3_path = os.path.expanduser('~/calibtration_images/rel_pos3.jpg')
rel_pos4_path = os.path.expanduser('~/calibtration_images/rel_pos4.jpg')
rel_pos5_path = os.path.expanduser('~/calibtration_images/rel_pos5.jpg')
rel_pos6_path = os.path.expanduser('~/calibtration_images/rel_pos6.jpg')
rel_pos7_path = os.path.expanduser('~/calibtration_images/rel_pos7.jpg')
rel_pos8_path = os.path.expanduser('~/calibtration_images/rel_pos8.jpg')
rel_pos9_path = os.path.expanduser('~/calibtration_images/rel_pos9.jpg')
rel_pos10_path = os.path.expanduser('~/calibtration_images/rel_pos10.jpg')
rel_pos11_path = os.path.expanduser('~/calibtration_images/rel_pos11.jpg')
rel_pos12_path = os.path.expanduser('~/calibtration_images/rel_pos12.jpg')
rel_pos13_path = os.path.expanduser('~/calibtration_images/rel_pos13.jpg')
rel_pos14_path = os.path.expanduser('~/calibtration_images/rel_pos14.jpg')
rel_pos15_path = os.path.expanduser('~/calibtration_images/rel_pos15.jpg')
rel_pos16_path = os.path.expanduser('~/calibtration_images/rel_pos16.jpg')
rel_pos17_path = os.path.expanduser('~/calibtration_images/rel_pos17.jpg')

image_list=[initial_image_path,
rel_pos0_path,
rel_pos1_path, #hier
rel_pos2_path,
rel_pos3_path,
rel_pos4_path, #hier
rel_pos5_path,
rel_pos6_path,
rel_pos7_path, #hier
rel_pos8_path, #hier
rel_pos9_path, #hier
rel_pos10_path, #hier
rel_pos11_path,
rel_pos12_path, #hier
rel_pos13_path, #hier
rel_pos14_path, #hier
rel_pos15_path, #hier
rel_pos16_path, #hier
rel_pos17_path
]
pattern_size = [9,6]
square_size = 0.025 # 25mm = 0.025m

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

objp = np.zeros((pattern_size[1]*pattern_size[0], 3), np.float32)
objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2) * square_size

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
flags = cv2.CALIB_USE_INTRINSIC_GUESS
flags += cv2.CALIB_ZERO_TANGENT_DIST 
#flags += cv2.CALIB_FIX_SKEW

image = cv2.imread(image_list[0])

height, width, channels = image.shape

camera_matrix = np.array(
	[[max(height, width),0,width/2],
	[0,max(height, width),height/2],
	[0,0,1]]
	)

print(camera_matrix)


for image in image_list:

	img = cv2.imread(image, flags=cv2.IMREAD_COLOR+cv2.IMREAD_IGNORE_ORIENTATION)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

	if ret is False:
		print('Couldn\'t find image corner within'+image)
		sys.exit()

	cv2.imshow("Gray", gray)
	cv2.waitKey()

	winSize = (11, 11)
	zeroZone = (-1, -1)

	objpoints.append(objp)

	corners2 = cv2.cornerSubPix(gray, corners, winSize, zeroZone, criteria)
	imgpoints.append(corners2)

	cv2.drawChessboardCorners(img, (9,6), corners2, ret)
	cv2.imshow('img', img)
	cv2.waitKey()

cv2.destroyAllWindows()
idx = pattern_size[0]-1

rv = cv2.calibrateCameraRO(objpoints, imgpoints, gray.shape[::-1], idx, camera_matrix, None, flags=flags)

ret, camera_matrix, dist, rvecs, tvecs, newObjPoints = rv[:6] 

calibration_data ={
	'RMS reprojection error' : ret,
	'Refined intrinsic matrix' : camera_matrix.tolist(),
	'Distortion coefficients[k1, k2, p1, p2, k3]' : dist.tolist(),
	'Rotation vectors' : [r.tolist() for r in rvecs],
	'Translation vectors' : [t.tolist() for t in tvecs],
	'Refined 3D object points' : newObjPoints.tolist()
}

print(calibration_data)

output_path = os.path.expanduser('~/calibtration_images/calibration_data_5.json')
with open(output_path, 'w', encoding='utf-8') as f:
	json.dump(calibration_data, f)

