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


class WF_Pose:
    # cartesian coordinates
    cart_coordinates = np.array([0.450628752997, 0.161615832271, 0.217447307078])
    # quaternion coordinates
    quart_coordinates = np.array([0.704020578925, 0.710172716916, 0.00244101361829, 0.00194372088834])
    def __init__(self, c_c, q_c):
        self.cart_coordinates = np.array(c_c)
        self.quart_coordinates = np.array(q_c)

class JF_Configuration:
    j_coord = np.array([0.7, 0.4, -1.7, 1.4, -1.1, -1.6, -0.4])
    names = ['right_j0', 'right_j1', 'right_j2', 'right_j3','right_j4', 'right_j5', 'right_j6']
    def __init__(self, joints):
        self.j_coord = np.array(joints)
        



def ik_get_joint_configuration(desired_position):
    rospy.logerr("Service started")
    ns = "ExternalTools/right/PositionKinematicsNode/IKService"
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest()
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    poses = {
        'right': PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point(
                    x=desired_position.cart_coordinates[0],
                    y=desired_position.cart_coordinates[1],
                    z=desired_position.cart_coordinates[2],
                ),
                orientation=Quaternion(
                    x=desired_position.quart_coordinates[0],
                    y=desired_position.quart_coordinates[1],
                    z=desired_position.quart_coordinates[2],
                    w=desired_position.quart_coordinates[3],
                ),
            ),
        ),
    }
    # Add desired pose for inverse kinematics
    ikreq.pose_stamp.append(poses['right'])
    # Request inverse kinematics from base to "right_hand" link
    ikreq.tip_names.append('right_hand')


    try:
        rospy.wait_for_service(ns, 5.0)
        resp = iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException) as e:
        rospy.logerr("Service call failed: %s" % (e,))
        return False

    if (resp.result_type[0] > 0):
        seed_str = {
                    ikreq.SEED_USER: 'User Provided Seed',
                    ikreq.SEED_CURRENT: 'Current Joint Angles',
                    ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                   }.get(resp.result_type[0], 'None')
        rospy.loginfo("SUCCESS - Valid Joint Solution Found from Seed Type: %s" %
              (seed_str,))
        # Format solution into Limb API-compatible dictionary
        limb_joints = dict(list(zip(resp.joints[0].name, resp.joints[0].position)))
        rospy.loginfo("\nIK Joint Solution:\n%s", limb_joints)
        rospy.loginfo("------------------")
        rospy.loginfo("Response Message:\n%s", resp)
    else:
        rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
        rospy.logerr("Result Error %d", resp.result_type[0])
        return False

    return limb_joints


def execute_JF_config(j_config, time_for_movement = 3.0, speed = 0.3, accuracy = 0.001):
    waypoint = {
        j_config.names[0] : j_config.j_coord[0],
        j_config.names[1] : j_config.j_coord[1],
        j_config.names[2] : j_config.j_coord[2],
        j_config.names[3] : j_config.j_coord[3],
        j_config.names[4] : j_config.j_coord[4],
        j_config.names[5] : j_config.j_coord[5],
        j_config.names[6] : j_config.j_coord[6]
        }

    limb = intera_interface.Limb('right')
    limb.set_joint_position_speed(speed)

    limb.move_to_joint_positions(waypoint, timeout=time_for_movement)

def gripper_config_callback(msg):
    """
    config topic callback
    """
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.data)

def get_right_hand_pose_WF(): #Work in Progress
    listener = tf.TransformListener()
    listener.waitForTransform('/base', '/right_hand', rospy.Time(0), rospy.Duration(1.0))


    (trans,rot) = listener.lookupTransform('/base', '/right_hand', rospy.Time(0))

    return WF_Pose(trans, rot)

def get_camera_pose_WF(): #Test required
    listener = tf.TransformListener()
    listener.waitForTransform('/base', '/right_hand_camera', rospy.Time(0), rospy.Duration(1.0))


    (trans,rot) = listener.lookupTransform('/base', '/right_hand_camera', rospy.Time(0))

    return WF_Pose(trans, rot)

def get_joint_configuration():
    #rospy.init_node("record_joint_pose")
    limb = intera_interface.Limb('right')
    input("Move the robot to a desired pose and press ENTER to record joint angles...")
    joints = limb.joint_angles()
    print("Recorded Pose:")
    j_config = [j_c for j, j_c in joints.items()]
    print(j_config)

    return JF_Configuration(j_config)

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


def take_calibration_images(start_pose): #Work in Progress
    relative_position_dict = {'rel_pos_1': {'cart': np.array([-0.0214327 , -0.08090044,  0.        ]), 'quart': np.array([ 0.0086425 , -0.01844621, -0.04233024, -0.00062285])},
                              'rel_pos_2': {'cart': np.array([-0.05807556,  0.09800256,  0.        ]), 'quart': np.array([ 0.00846041, -0.00371431,  0.01000697,  0.01288309])},
                              'rel_pos_3': {'cart': np.array([-0.06155737,  0.13584084,  0.        ]), 'quart': np.array([-0.00356379, -0.03062105, -0.0244527 ,  0.00922645])}, 
                              'rel_pos_4': {'cart': np.array([-0.04373173, -0.08075334, 0.]), 'quart': np.array([ 0.04885371, -0.03413394, -0.09992369,  0.00404833])}, 
                              'rel_pos_5': {'cart': np.array([-0.03286316,  0.12535362,  0.        ]), 'quart': np.array([-0.12181704, -0.16448839, -0.18990887, -0.10695132])}, 
                              'rel_pos_6': {'cart': np.array([0.12765851, 0.01223566, 0.        ]), 'quart': np.array([ 0.09637976,  0.18602444,  0.291096  , -0.04709941])}, 
                              'rel_pos_7': {'cart': np.array([0.06732954,  -0.02254284, 0.]), 'quart': np.array([0.22413376, -0.18579842,  0.19845549, -0.26756999])}, 
                              'rel_pos_8': {'cart': np.array([0.05570808, 0.00906466, 0.        ]), 'quart': np.array([0.24382643, 0.18214966, 0.35641839, 0.01774297])}, 
                              'rel_pos_9': {'cart': np.array([-0.03218547, -0.05974087,  0.        ]), 'quart': np.array([ 0.32412512, 0.12740648, 0.35653082, 0.0990276])},
                              'rel_pos_10': {'cart': np.array([-0.03756981, -0.03590446,  0.        ]), 'quart': np.array([ 0.11319734, 0.06868901, 0.10168174, 0.0499594])}, 
                              'rel_pos_11': {'cart': np.array([-0.01687394, -0.02549765,  0.        ]), 'quart': np.array([ 0.10481816, 0.0478516 , 0.0964062 , 0.06150413])}, 
                              'rel_pos_12': {'cart': np.array([0.01716224, -0.05288254,  0.        ]), 'quart': np.array([ 0.119263  , 0.0546035 , 0.12111945, 0.07019976])},
                              'rel_pos_13': {'cart': np.array([-0.0671396 ,  0.07953471,  0.        ]), 'quart': np.array([ 0.22268461, 0.0838357 , 0.2379888 , 0.10813681])}, 
                              'rel_pos_14': {'cart': np.array([ 0.04152203, -0.17244348,  0.        ]), 'quart': np.array([0.32684166, 0.16900653, 0.53686025, 0.0122754 ])}, 
                              'rel_pos_15': {'cart': np.array([0.03034981, 0.03632107, 0.        ]), 'quart': np.array([0.15746242, 0.12360467, 0.44522665, 0.03113727])}}

    initial_pose = WF_Pose([start_pose.cart_coordinates[0],start_pose.cart_coordinates[1],0.20159622], start_pose.quart_coordinates)
    joint_angles = ik_get_joint_configuration(initial_pose)
    if joint_angles:
        rospy.loginfo("Joint angles: %s", joint_angles)
    else:
        rospy.logerr("No valid joint configuration found.")

    liste = [joint_angles[joint] for joint in joint_angles]
    Joint_Configuration = JF_Configuration(liste)

    execute_JF_config(Joint_Configuration,7, 0.2, 0.001)

    img = extract_single_image("right_hand_camera")

    output_path = os.path.expanduser('~/calibtration_images/initial_image.jpg')
    if cv2.imwrite(output_path, img):
        rospy.loginfo(f"Bild erfolgreich gespeichert unter: {output_path}")
    else:
        rospy.logerr(f"Fehler beim Speichern des Bildes unter: {output_path}")
    for i, relpos in enumerate(relative_position_dict):
        cart_cord = initial_pose.cart_coordinates+relative_position_dict[relpos]['cart']
        quart_cord = initial_pose.quart_coordinates+relative_position_dict[relpos]['quart']
        relPose = WF_Pose(cart_cord, quart_cord)
        joint_angles = ik_get_joint_configuration(relPose)
        if joint_angles:
            rospy.loginfo("Joint angles: %s", joint_angles)
        else:
            rospy.logerr("No valid joint configuration found.")

        liste = [joint_angles[joint] for joint in joint_angles]
        Joint_Configuration = JF_Configuration(liste)

        execute_JF_config(Joint_Configuration,7, 0.2, 0.001)

        img = extract_single_image("right_hand_camera")

        output_path = os.path.expanduser('~/calibtration_images/rel_pos'+str(i)+'.jpg')
        if cv2.imwrite(output_path, img):
            rospy.loginfo(f"Bild erfolgreich gespeichert unter: {output_path}")
        else:
            rospy.logerr(f"Fehler beim Speichern des Bildes unter: {output_path}")

def generate_relative_dictionary(start, list_rel_pos):
    dict_rel_pos ={}
    for i in range(len(list_rel_pos)):
        dict_rel_pos['rel_pos'+str(i+1)]={}
        dict_rel_pos['rel_pos'+str(i+1)]['cart']=list_rel_pos[i].cart_coordinates-start.cart_coordinates
        dict_rel_pos['rel_pos'+str(i+1)]['quart']=list_rel_pos[i].quart_coordinates-start.quart_coordinates
    print(dict_rel_pos)

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
        ratio = minEllipse[1][0]/minEllipse[1][1]

        if ratio < 0.69 or ratio > 0.75:
            continue

            
        ellipse_list.append(minEllipse)
        print(minEllipse[1][0])
        print(minor)
        color = (0, 255, 0)
        cv2.ellipse(image, minEllipse, color, 2)
        
    # show the output image
    imS = cv2.resize(image, (960, 540))
    cv2.imshow("Output", imS)
    cv2.waitKey(0)
    print(ellipse_list)
    return ellipse_list, newcameramatrix, calibration_coefficients

def get_egg_coordinates_in_camera_coordinate_system(pixel_coordinates, intrinsic_camera_matrix, Z_c):
    cont_pixel_coordinates = np.array([pixel_coordinates[0], pixel_coordinates[1], 1])
    inverse_i_c_m = np.linalg.inv(intrinsic_camera_matrix)

    c_c= np.matmul(inverse_i_c_m, cont_pixel_coordinates) * Z_c

    return c_c


def get_gripper_angle():
    # Do te angle calculation by means of a unit normal vector \overline{x}=[cos(\alpha), sin(\alpha)]
    pass

    


#extracting camera pose
#rospy.Subscriber('io/internal_camera/config', IONodeConfiguration, self._node_config_cb)


if __name__ == '__main__':
    
    #destination = WF_Pose([0.50664216, -0.01078819, 0.24],[0.0, 1.0, 0.0, 0.0])
    # quat orientation(camera orientation) camera almost paralell: -0.79868572  0.60128082 -0.00647855  0.02281495 
    # quat orientation(right hand orientation) camera almost paralell: 0.38973709 0.56902529 0.55582616 0.46408242

    #-1.0412168
    initial_joint_configuration = JF_Configuration([0.7780993895639919, 0.682433800916642, 0.3084028103250309, -1.1754058079678202, -0.24632390895499304, 0.4452962076276255, -2.9687903531269133])
    #over_eggs_joint_configuration = JF_Configuration([ 0.73041406, 0.60819629, 0.17981836, -0.97856934, -0.11923926, 0.36015039, 0 ])
    over_eggs_joint_configuration = JF_Configuration([0.553681640625, 0.615490234375, 0.1306494140625, -1.6195478515625, -0.1760732421875, 0.9588955078125, 2.567326171875])
    Z_c = 0.5687302672979888 #Please always update 0.5726654612345957  first trial 0.34316552906531683
    h_neg = Z_c - 17.5


    rospy.init_node("rsdk_ik_service_client")

    start_pose = WF_Pose([0.70334749, 0.60949191, 0.18480601],[0.25651805, 0.65661396, 0.52851486, 0.47299963]) 


    initial_position = WF_Pose([0.81062593, 0.43855336, 0.19504099],[-0.07170088,  0.72394554,  0.03336781,  0.685309]
)

    destination1 = WF_Pose([0.70286178, 0.40400799, 0.20159622],[-0.43830479, 0.50061232, -0.3434631, 0.66280413])
    
    destination2 = WF_Pose([0.66621892,0.58291099, 0.20159622],[-0.43848688, 0.51534422, -0.29112589, 0.67631007])

    destination3 = WF_Pose([0.66273711, 0.62074927, 0.20159622],[-0.45051108, 0.48843748, -0.32558556, 0.67265343])


    destination4 = WF_Pose([0.68056275, 0.40415509, 0.1738608],[-0.39809358, 0.48492459, -0.40105655, 0.66747531])

    destination5 = WF_Pose([0.69143132, 0.61026205, 0.20159622],[-0.56876433, 0.35457014, -0.49104173, 0.55647566])

    destination6 = WF_Pose([0.85195299, 0.49714409, 0.20159622],[-0.35056753, 0.70508297, -0.01003686, 0.61632757])


    #to be reworked

    destination7 = WF_Pose([0.72429448, 0.48490843, 0.20159622],[-0.44694729, 0.51905853, -0.30113286, 0.66342698])

    

    destination8 = WF_Pose([0.78000256, 0.49397309, 0.20159622],[-0.20312086, 0.70120819, 0.05528553, 0.68116995])

    destination9 = WF_Pose([0.70517375, 0.42348438, 0.21365166],[-0.12838385, 0.65211825,  0.03854294, 0.74617278])

    destination10 = WF_Pose([0.69978941, 0.44732079, 0.20931623],[-0.33931163, 0.59340078, -0.21630614, 0.69710458])

    destination11 = WF_Pose([0.72048528, 0.4577276, 0.20912388],[-0.34769081, 0.57256337, -0.22158168, 0.70864931])

    destination12 = WF_Pose([0.75452146, 0.43034271, 0.22773396],[-0.33324597, 0.57931527, -0.19686843, 0.71734494])

    destination13 = WF_Pose([0.67021962, 0.56275996, 0.2102942],[-0.22982436, 0.60854747, -0.07999908, 0.75528199])

    destination14 = WF_Pose([0.76581651, 0.31246495, 0.20159622],[-0.12010563, 0.68806506, 0.23572739, 0.67570238])


    destination15 = WF_Pose([0.75464429, 0.5212295, 0.20159622],[-0.28948487,  0.6426632,   0.14409379,  0.69456425])

    
    #rel_pos14_path, #hier
    #rel_pos15_path, #hier
    #rel_pos16_path, #hier
    
    

    #j_config = get_joint_configuration()

    #print(j_config.names)
    #print(j_config.j_coord)

    #execute_JF_config(over_eggs_joint_configuration,3, 0.2, 0.001)
    '''
    egg_image = extract_single_image("right_hand_camera")
    output_path = os.path.expanduser('~/calibtration_images/egg.jpg')
    if cv2.imwrite(output_path, egg_image):
        rospy.loginfo(f"Bild erfolgreich gespeichert unter: {output_path}")
    else:
        rospy.logerr(f"Fehler beim Speichern des Bildes unter: {output_path}")

    ellipses, camera_matrix, dist_coeff = watershed_ellipse_detection(egg_image)

    print(ellipses[0][0])
    print(camera_matrix)
    '''
    # start from here
    
    middle_joints = JF_Configuration([1.2609267578125, 0.6007939453125, 1.1895029296875, -1.991818359375, -1.1004970703125, 0.29201953125, 2.567326171875])
    final_joints = JF_Configuration([0.791490234375, 0.566357421875, 1.406859375, -1.411169921875, -1.30523828125, 0.1868857421875, 1.9222822265625])
    #3,2, mid,final to reach egg box

    #hallo = get_joint_configuration()



    get_joint_configuration()

    execute_JF_config(over_eggs_joint_configuration,10, 0.2, 0.001)
    

    
    

    
    
    gripper = intera_interface.Gripper()

    gripper.open()
    execute_JF_config(over_eggs_joint_configuration,10, 1.3, 0.001)

    egg_image = extract_single_image("right_hand_camera")

    ellipses, camera_matrix, dist_coeff = watershed_ellipse_detection(egg_image)

    for ellipse in ellipses:
        ratio = ellipse[1][0]/ellipse[1][1]
        print('Ratio:')
        print(ratio)

    egg_position = ellipses[0][0]

    egg_angle = ellipses[0][2]
    print(egg_angle)

    egg_angle_radians = np.deg2rad(egg_angle)

    wf_camera = get_camera_pose_WF()
    wf_right_hand = get_right_hand_pose_WF()

    unit_vector_egg_angle = np.array([np.cos(egg_angle_radians),np.sin(egg_angle_radians), 0 ])
    unit_vector_egg_angle_cf = np.array([unit_vector_egg_angle[0], unit_vector_egg_angle[1], 0]) 

    print('right hand camera pose')
    print(wf_camera.cart_coordinates)
    print(wf_camera.quart_coordinates)

    print('right hand pose')
    print(wf_right_hand.cart_coordinates)
    print(wf_right_hand.quart_coordinates)

    intrinsic_camera_matrix = np.array([[620.84861718,   0.,         365.72552218],
 [  0.,         623.17465125, 217.45728018],
 [  0.,           0.,           1.        ]])
    
    print(intrinsic_camera_matrix)


    quat = Rotation.from_quat(wf_camera.quart_coordinates)
    euler_angles = quat.as_euler('xyz', degrees= True)
    print(euler_angles)

    x = get_egg_coordinates_in_camera_coordinate_system(egg_position, intrinsic_camera_matrix, Z_c)

    print(x)

    x_base = np.array(quat.apply(x))
    unit_vector_egg_angle_cf = np.array(quat.apply(unit_vector_egg_angle_cf))

    x_y_translation = np.array([x_base[0], x_base[1], 0])

    print(x_base)

    #get_joint_configuration()



    translation_coordinates = wf_camera.cart_coordinates+x_y_translation
    almost_final_angle = np.arctan2(unit_vector_egg_angle_cf[1], unit_vector_egg_angle_cf[0])
    final_angle = np.rad2deg(almost_final_angle)+90

    rot_target = Rotation.from_euler('xyz', [180.0, 0.0, final_angle], degrees= True)
    quat_target = rot_target.as_quat()

    rot = Rotation.from_quat([0.0, 1.0, 0.0, 0.0])
    rot_euler = rot.as_euler('xyz', degrees= True)

    
    # Go to cetrtain position
    pose_over_egg = WF_Pose(translation_coordinates, quat_target)

    joint_angles_over_egg = ik_get_joint_configuration(pose_over_egg)
    if joint_angles_over_egg:
        rospy.loginfo("Joint angles: %s", joint_angles_over_egg)
    else:
        rospy.logerr("No valid joint configuration found.")


    liste = [joint_angles_over_egg[joint] for joint in joint_angles_over_egg]
    print(liste)

    Joint_Configuration = JF_Configuration(liste)

    execute_JF_config(Joint_Configuration,10, 1.3, 0.001)


    wf_right_hand = get_right_hand_pose_WF()

    approach_egg = WF_Pose([wf_right_hand.cart_coordinates[0],wf_right_hand.cart_coordinates[1],wf_right_hand.cart_coordinates[2]-h_neg], 
                           wf_right_hand.quart_coordinates)
    
    joint_angles = ik_get_joint_configuration(approach_egg)
    if joint_angles:
        rospy.loginfo("Joint angles: %s", joint_angles)
    else:
        rospy.logerr("No valid joint configuration found.")


    liste = [joint_angles[joint] for joint in joint_angles]

    Joint_Configuration_Approach = JF_Configuration(liste)

    execute_JF_config(Joint_Configuration_Approach,10, 0.1, 0.001)

    

    gripper.close()

    while gripper.is_moving():
        rospy.sleep(0.5)

    execute_JF_config(joint_angles_over_egg,10, 0.2, 0.001)

    
    execute_JF_config(middle_joints,10, 1.3, 0.001)

    execute_JF_config(final_joints,10, 0.2, 0.001)

    gripper.open()

    while gripper.is_moving():
        rospy.sleep(0.5)

    execute_JF_config(over_eggs_joint_configuration,10, 0.2, 0.001)


    
    







   




    

    
    
    
    