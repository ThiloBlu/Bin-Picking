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
input_path = os.path.expanduser('~/calibtration_images/calibration_data_1.json')
with open(input_path, 'r', encoding='utf-8') as f:
	calibration_parameters1 = json.load(f)
	
intrinsic_matrix_1 = np.array(calibration_parameters1['Refined intrinsic matrix'])
calibration_coefficients_1 = np.array(calibration_parameters1['Distortion coefficients[k1, k2, p1, p2, k3]'])

input_path = os.path.expanduser('~/calibtration_images/calibration_data_2.json')
with open(input_path, 'r', encoding='utf-8') as f:
	calibration_parameters2 = json.load(f)
	
intrinsic_matrix_2 = np.array(calibration_parameters2['Refined intrinsic matrix'])
calibration_coefficients_2 = np.array(calibration_parameters2['Distortion coefficients[k1, k2, p1, p2, k3]'])

input_path = os.path.expanduser('~/calibtration_images/calibration_data_3.json')
with open(input_path, 'r', encoding='utf-8') as f:
	calibration_parameters3 = json.load(f)
	
intrinsic_matrix_3 = np.array(calibration_parameters3['Refined intrinsic matrix'])
calibration_coefficients_3 = np.array(calibration_parameters3['Distortion coefficients[k1, k2, p1, p2, k3]'])

input_path = os.path.expanduser('~/calibtration_images/calibration_data_4.json')
with open(input_path, 'r', encoding='utf-8') as f:
	calibration_parameters4 = json.load(f)
	
intrinsic_matrix_4 = np.array(calibration_parameters4['Refined intrinsic matrix'])
calibration_coefficients_4 = np.array(calibration_parameters4['Distortion coefficients[k1, k2, p1, p2, k3]'])

input_path = os.path.expanduser('~/calibtration_images/calibration_data_5.json')
with open(input_path, 'r', encoding='utf-8') as f:
	calibration_parameters5 = json.load(f)
	
intrinsic_matrix_5 = np.array(calibration_parameters5['Refined intrinsic matrix'])
calibration_coefficients_5 = np.array(calibration_parameters5['Distortion coefficients[k1, k2, p1, p2, k3]'])

intrinsic_matrix_final = (intrinsic_matrix_1+intrinsic_matrix_2+ intrinsic_matrix_3+intrinsic_matrix_4+intrinsic_matrix_5)/5
calibration_coefficients_final = (calibration_coefficients_1+ calibration_coefficients_2+calibration_coefficients_3+calibration_coefficients_4+calibration_coefficients_5)/5

print(intrinsic_matrix_1)
print(calibration_coefficients_1)

print(intrinsic_matrix_final)
print(calibration_coefficients_final)
'''
input_path = os.path.expanduser('~/calibtration_images/calibration_data_final.json')
with open(input_path, 'r', encoding='utf-8') as f:
	calibration_parameters = json.load(f)
	
intrinsic_matrix = np.array(calibration_parameters['Refined intrinsic matrix'])
calibration_coefficients = np.array(calibration_parameters['Distortion coefficients[k1, k2, p1, p2, k3]'])

#egg.jpg
#initial_image

initial_image_path = os.path.expanduser('~/calibtration_images/determine_height.jpg')
img = cv2.imread(initial_image_path)
h,  w = img.shape[:2]
newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(intrinsic_matrix, calibration_coefficients, (w,h), 1, (w,h))

print(newcameramatrix)

calibration_data ={
	'Undistorted intrinsic camera matrix' : newcameramatrix.tolist(),
}

print(calibration_data)

output_path = os.path.expanduser('~/calibtration_images/undistorted_intrinsic_camera_matrix.json')
with open(output_path, 'w', encoding='utf-8') as f:
	json.dump(calibration_data, f)

# undistort
undistorted_image = cv2.undistort(img, intrinsic_matrix, calibration_coefficients, None, newcameramatrix)

print(newcameramatrix[0][0])
print(newcameramatrix[1][1])
 
# crop the image
x, y, w, h = roi
undistorted_image = undistorted_image[y:y+h, x:x+w]
calibrated_initial_image_path = os.path.expanduser('~/calibtration_images/calibrated_initial_image.jpg')
cv2.imwrite(calibrated_initial_image_path, undistorted_image)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
pattern_size = [9,6]
img = cv2.imread(calibrated_initial_image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)


cv2.imshow("Gray", gray)
cv2.waitKey()

winSize = (11, 11)
zeroZone = (-1, -1)

corners2 = cv2.cornerSubPix(gray, corners, winSize, zeroZone, criteria)

cv2.drawChessboardCorners(img, (9,6), corners2, ret)
cv2.imshow('img', img)
output_path = os.path.expanduser('~/calibtration_images/determine_height_corners.jpg')
if cv2.imwrite(output_path, img):
	print(f"Bild erfolgreich gespeichert unter: {output_path}")
else:
	print(f"Fehler beim Speichern des Bildes unter: {output_path}")

cv2.waitKey()



first_point = np.array(corners[0])
second_point = np.array(corners[9])

Z_c = (newcameramatrix[1][1]+newcameramatrix[0][0])/(2) * 0.025 /(np.sqrt(np.power((first_point[0][0]-second_point[0][0]),2)+np.power((first_point[0][1]-second_point[0][1]),2)))  
Z_cmax = newcameramatrix[1][1] * 0.025 /(np.sqrt(np.power((first_point[0][0]-second_point[0][0]),2)+np.power((first_point[0][1]-second_point[0][1]),2)))  
Z_cmin = newcameramatrix[0][0] * 0.025 /(np.sqrt(np.power((first_point[0][0]-second_point[0][0]),2)+np.power((first_point[0][1]-second_point[0][1]),2)))  

print(Z_c)
print(Z_cmax-Z_c)
print(Z_cmin - Z_c)

#0.34316552906531683
#0.5726654612345957
#0.5687302672979888

