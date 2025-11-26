#!/usr/bin/env python3
import rospy
import numpy as np
import os
import json
import time

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header
from sensor_msgs.msg import JointState

from intera_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)
import intera_interface
from intera_core_msgs.msg import IONodeConfiguration

import tf
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from intera_io import IODeviceInterface
from intera_interface.robot_params import RobotParams
from scipy.spatial.transform import Rotation
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import imutils


def extract_single_image(camera_name): #Test required
    intera_interface.Cameras().start_streaming(camera_name)
    cameras_io = dict()
    camera_param_dict = RobotParams().get_camera_details()
    camera_list = list(camera_param_dict.keys())

    rospy.loginfo("Warte 2 Sekunden, um das Bild zu stabilisieren...")
    rospy.sleep(2)

    camera_capabilities = {
        "mono": ['cognex'],
        "color": ['ienso_ethernet'],
        "auto_exposure": ['ienso_ethernet'],
        "auto_gain": ['ienso_ethernet']
    }
    for camera in camera_list:
        cameraType = camera_param_dict[camera]['cameraType']
        try:
            interface = IODeviceInterface("internal_camera", camera)

            cameras_io[camera] = {
                'interface': interface,
                'is_color': (cameraType in camera_capabilities['color']),
                'has_auto_exposure': (cameraType in camera_capabilities['auto_exposure']),
                'has_auto_gain': (cameraType in camera_capabilities['auto_gain']),
            }
        except OSError as e:
            rospy.logerr("Could not find expected camera ({0}) for this robot.\n"
                "Please contact Rethink support: support@rethinkrobotics.com".format(camera))

    if cameras_io[camera_name]['is_color']:
        image_string = "image_rect_color"
    else:
        image_string = "image_rect"
    try:
        for _ in range(3):
            _ = rospy.wait_for_message(f"/io/internal_camera/{camera_name}/{image_string}", Image)
        img_data = rospy.wait_for_message('/'.join(["/io/internal_camera", camera_name,
                    image_string]),  Image, timeout=1.0)
        time.sleep(0.05)
        rospy.loginfo(f"Bild von Kamera {camera_name} erfolgreich empfangen.")

    except rospy.ROSException as e:
        rospy.logerr(f"Fehler beim Empfangen des Bildes: {e}")
        return None

    bridge = CvBridge()
    try:
        cv_image_final = bridge.imgmsg_to_cv2(img_data, "bgr8")
        rospy.loginfo("Bild erfolgreich in OpenCV-Format konvertiert.")
    except CvBridgeError as err:
        rospy.logerr(f"Fehler bei der Bildkonvertierung: {err}")
        return None


    cv2.imshow(camera_name, cv_image_final)
    cv2.waitKey(3)

    return cv_image_final


if __name__ == '__main__':

    rospy.init_node("take_photos")

    img = extract_single_image("right_hand_camera")

    output_path = os.path.expanduser('~/calibtration_images/determine_height.jpg')
    if cv2.imwrite(output_path, img):
        rospy.loginfo(f"Bild erfolgreich gespeichert unter: {output_path}")
    else:
        rospy.logerr(f"Fehler beim Speichern des Bildes unter: {output_path}")