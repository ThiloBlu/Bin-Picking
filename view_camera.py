#!/usr/bin/env python3
import rospy
import numpy as np
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


def show_image_callback(img_data, window_name):
    """The callback function to show image by using CvBridge and cv
    """
    rospy.loginfo("Received image callback")

    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(img_data, "bgr8")
    except CvBridgeError as err:
        rospy.logerr(err)
        return

    cv2.namedWindow(window_name, 0)
    # refresh the image on the screen
    cv2.imshow(window_name, cv_image)
    cv2.waitKey(3)


def get_camera_images(camera_name): #Test required
    rp = intera_interface.RobotParams()
    valid_cameras = rp.get_camera_names()
    if not valid_cameras:
        rp.log_message(("Cannot detect any camera_config"
            " parameters on this robot. Exiting."), "ERROR")
        return

    
    cameras = intera_interface.Cameras()
    cameras.start_streaming(camera_name)

    rospy.loginfo(f"Setting callback for camera: {camera_name}")

    rectify_image = True

    cameras.set_callback(camera_name, show_image_callback,
        rectify_image=rectify_image, callback_args=camera_name)
    
if __name__ == '__main__':


    rospy.init_node("camera_display")


    get_camera_images("right_hand_camera")
    rospy.loginfo("Camera_display node running. Ctrl-c to quit")
    rospy.spin()