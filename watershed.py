#!/usr/bin/env python3
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import argparse
import imutils
import cv2
import os
import json

'''
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
'''
def watershed_ellipse_detection(img):
    #load intrinsic camera matrix and calibration coefficients
    input_path = os.path.expanduser('~/calibtration_images/calibration_data_final.json')
    with open(input_path, 'r', encoding='utf-8') as f:
        calibration_parameters = json.load(f)
        
    intrinsic_matrix = np.array(calibration_parameters['Refined intrinsic matrix'])
    calibration_coefficients = np.array(calibration_parameters['Distortion coefficients[k1, k2, p1, p2, k3]'])
    # load the image and perform pyramid mean shift filtering
    # to aid the thresholding step
    
    h,  w = img.shape[:2]
    newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(intrinsic_matrix, calibration_coefficients, (w,h), 1, (w,h))

    image = cv2.undistort(img, intrinsic_matrix, calibration_coefficients, None, newcameramatrix)
    
    # crop the image
    x, y, w, h = roi
    image = image[y:y+h, x:x+w]

    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
    cv2.imshow("Shifted", shifted)
    cv2.waitKey()
    # convert the mean shift image to grayscale, then apply
    # Otsu's thresholding
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Thresh", thresh)
    cv2.waitKey()

    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, min_distance=20,
        labels=thresh)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers_mask = np.zeros(D.shape, dtype=bool)
    markers_mask[tuple(localMax.T)]=True
    markers = ndimage.label(markers_mask, structure=np.ones((3, 3)))[0]

    print(localMax)
    print(markers_mask)

    labels = watershed(-D, markers, mask=thresh)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

    # loop over the unique labels returned by the Watershed
    # algorithm
    ellipse_list = []

    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")

        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        c = max(cnts, key=cv2.contourArea)

        if c.shape[0]<5:
            continue

        # draw a circle enclosing the object
        minEllipse = cv2.fitEllipse(c)
        major = minEllipse[1][0]
        minor = minEllipse[1][1]

        if major > 100:
            continue
        if minor > 120: 
            continue

            
        ellipse_list.append(minEllipse)
        print(minEllipse[1][0])
        print(minor)
        color = (0, 255, 0)
        cv2.ellipse(image, minEllipse, color, 2)
        cv2.putText(image, "{}".format(label), (int(minEllipse[0][0])-10 , int(minEllipse[0][1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    # show the output image
    imS = cv2.resize(image, (960, 540))
    cv2.imshow("Output", imS)
    cv2.waitKey(0)
    print(ellipse_list)
    return ellipse_list, newcameramatrix, calibration_coefficients

if __name__ == '__main__':
    image_path = os.path.expanduser('~/calibtration_images/egg.jpg')
    img = cv2.imread(image_path) # hier abändern für bin_picking
    ellipses, camera_matrix, dist_coeff = watershed_ellipse_detection(img)

    print(ellipses[0][0])
    print(camera_matrix)