import cv2
import math
import numpy as np
from numpy import cos,sin
import pandas as pd
import pykinect_azure as pykinect
from openpyxl import Workbook
import csv
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion import Quaternion as pq
import ctypes
from pykinect_azure.k4abt._k4abtTypes import k4a_float3_t, k4a_quaternion_t

def get_unit_quat_and_transformation_matrix(body_frame, joint_type):
    body = body_frame.get_body(0)
    joint_info = body.joints[joint_type].orientation
    quat_dict = joint_info.__iter__()
    vector_4d = np.array([quat_dict['w'], quat_dict['x'], quat_dict['y'], quat_dict['z']])
    quat = pq(array=vector_4d)
    unit_quat = quat.normalised
    transformation_matrix = quat.transformation_matrix
    return unit_quat, transformation_matrix
def twoD_to_threeD(color_skeleton_2d, depth_skeleton_2d,depth_image, device, calibration, joint):
    color_joint_2d = color_skeleton_2d[joint,:]
    depth_joint_2d = depth_skeleton_2d[joint,:]
    
    depth_float2 = pykinect.k4a_float2_t(depth_joint_2d)
    depth = depth_image[int(depth_joint_2d[1]), int(depth_joint_2d[0])]
    depth_float3 = device.calibration.convert_2d_to_3d(depth_float2,depth, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
    depth_transformed_3d = [depth_float3.xyz.x, depth_neck_float3.xyz.y, depth_neck_float3.xyz.z]
    
    color_joint_3d = transformed_points_map[int(color_joint_2d[1]), int(color_joint_2d[0]), :]
    depth_joint_3d =  points_map[int(depth_joint_2d[1]), int(depth_joint_2d[0]), :]
    joint_3d = skeleton_3d[joint,:3]
			
    return depth_transformed_3d, color_joint_3d, depth_joint_3d, joint_3d
    
def dh_matrix(a_val, alpha_val, d_val, theta_val):
    """
    Calculate the DH transformation matrix given DH parameters.

    Parameters:
    a_val (float): Link length.
    alpha_val (float): Link twist in radians.
    d_val (float): Link offset.
    theta_val (float): Joint angle in radians.

    Returns:
    numpy.ndarray: DH transformation matrix.
    """
    matrix = np.array([
        [cos(theta_val), -sin(theta_val) , 0, a_val]
        [sin(theta_val)*cos(alpha_val), cos(theta_val) * cos(alpha_val), -sin(alpha_val), -sin(alpha_val)*d_val],
        [sin(theta_val)*sin(alpha_val), cos(theta_val)*sin(alpha_val), cos(alpha_val), cos(alpha_val)*d_val],
        [0, 0, 0, 1]
    ])
    return matrix


    

if __name__ == "__main__":

	# Initialize the library, if the library is not found, add the library path as argument
	pykinect.initialize_libraries(track_body=True)

	# Modify camera configuration
	device_config = pykinect.default_configuration
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
	device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
	

	# Start device
	device = pykinect.start_device(config=device_config)

	# Start body tracker
	bodyTracker = pykinect.start_body_tracker()

	cv2.namedWindow('Color image with skeleton',cv2.WINDOW_NORMAL)
	list_degrees_z_list = []
	list_degrees_y_list = []
	list_degrees_x_list = []
	max_values = 5
	average_angle_z =0
	average_angle_x =0
	average_angle_y =0
	iteration = 0
	

	while True:
		iteration +=1
		
		# Get capture
		capture = device.update()

		# Get body tracker frame
		body_frame = bodyTracker.update()

		# Get the color image
		ret_color, color_image = capture.get_color_image()

		# Get the depth image
		ret_depth, depth_image = capture.get_depth_image()

		# Get the transformed color image
		ret_transformed_color, transformed_color_image = capture.get_transformed_color_image()

		# Get the point cloud
		ret_point, points = capture.get_pointcloud()

		# Get the transformed point cloud
		ret_transformed_point, transformed_points = capture.get_transformed_pointcloud()

		if not ret_color or not ret_depth or not ret_point or not ret_transformed_point or not ret_transformed_color:
			continue

		points_map = points.reshape((transformed_color_image.shape[0], transformed_color_image.shape[1], 3))
		transformed_points_map = transformed_points.reshape((color_image.shape[0], color_image.shape[1], 3))
  
		

		for body_id in range(body_frame.get_num_bodies()):
			color_skeleton_2d = body_frame.get_body2d(body_id, pykinect.K4A_CALIBRATION_TYPE_COLOR).numpy()
			depth_skeleton_2d = body_frame.get_body2d(body_id, pykinect.K4A_CALIBRATION_TYPE_DEPTH).numpy()
			skeleton_3d = body_frame.get_body(body_id).numpy()
			#left
			color_neck_2d = color_skeleton_2d[pykinect.K4ABT_JOINT_NECK,:]
			depth_neck_2d = depth_skeleton_2d[pykinect.K4ABT_JOINT_NECK,:]
   
			color_left_clavicle_2d = color_skeleton_2d[pykinect.K4ABT_JOINT_CLAVICLE_LEFT,:]
			depth_left_clavicle_2d = depth_skeleton_2d[pykinect.K4ABT_JOINT_CLAVICLE_LEFT,:]

			color_left_shoulder_2d = color_skeleton_2d[pykinect.K4ABT_JOINT_SHOULDER_LEFT,:]
			depth_left_shoulder_2d = depth_skeleton_2d[pykinect.K4ABT_JOINT_SHOULDER_LEFT,:]
   
			#color_left_elbow_2d = color_skeleton_2d[pykinect.K4ABT_JOINT_ELBOW_LEFT,:]
			#depth_left_elbow_2d = depth_skeleton_2d[pykinect.K4ABT_JOINT_ELBOW_LEFT,:]
   
			#color_left_wrist_2d = color_skeleton_2d[pykinect.K4ABT_JOINT_WRIST_LEFT,:]
			#depth_left_wrist_2d = depth_skeleton_2d[pykinect.K4ABT_JOINT_WRIST_LEFT,:]
   
			color_spine_chest_2d = color_skeleton_2d[pykinect.K4ABT_JOINT_SPINE_CHEST,:]
			depth_spine_chest_2d = depth_skeleton_2d[pykinect.K4ABT_JOINT_SPINE_CHEST,:]
	
			color_spine_navel_2d = color_skeleton_2d[pykinect.K4ABT_JOINT_SPINE_NAVEL,:]
			depth_spine_navel_2d = depth_skeleton_2d[pykinect.K4ABT_JOINT_SPINE_NAVEL,:]



			depth_neck_float2 = pykinect.k4a_float2_t(depth_neck_2d)
			depth = depth_image[int(depth_neck_2d[1]), int(depth_neck_2d[0])]
			depth_neck_float3 = device.calibration.convert_2d_to_3d(depth_neck_float2, depth, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
			depth_transformed_neck_3d = [depth_neck_float3.xyz.x, depth_neck_float3.xyz.y, depth_neck_float3.xyz.z]
   
			depth_left_clavicle_float2 = pykinect.k4a_float2_t(depth_left_clavicle_2d)
			depth = depth_image[int(depth_left_clavicle_2d[1]), int(depth_left_clavicle_2d[0])]
			depth_left_clavicle_float3 = device.calibration.convert_2d_to_3d(depth_left_clavicle_float2, depth, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
			depth_transformed_left_clavicle_3d = [depth_left_clavicle_float3.xyz.x, depth_left_clavicle_float3.xyz.y, depth_left_clavicle_float3.xyz.z]
   
			depth_left_shoulder_float2 = pykinect.k4a_float2_t(depth_left_shoulder_2d)
			depth = depth_image[int(depth_left_shoulder_2d[1]), int(depth_left_shoulder_2d[0])]
			depth_left_shoulder_float3 = device.calibration.convert_2d_to_3d(depth_left_shoulder_float2, depth, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
			depth_transformed_left_shoulder_3d = [depth_left_shoulder_float3.xyz.x, depth_left_shoulder_float3.xyz.y, depth_left_shoulder_float3.xyz.z]
   
			#depth_left_elbow_float2 = pykinect.k4a_float2_t(depth_left_elbow_2d)
			#depth = depth_image[int(depth_left_elbow_2d[1]), int(depth_left_elbow_2d[0])]
			#depth_left_elbow_float3 = device.calibration.convert_2d_to_3d(depth_left_elbow_float2, depth, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
			#depth_transformed_left_elbow_3d = [depth_left_elbow_float3.xyz.x, depth_left_elbow_float3.xyz.y, depth_left_elbow_float3.xyz.z]
   
			#depth_left_wrist_float2 = pykinect.k4a_float2_t(depth_left_wrist_2d)
			#depth = depth_image[int(depth_left_wrist_2d[1]), int(depth_left_wrist_2d[0])]
			#depth_left_wrist_float3 = device.calibration.convert_2d_to_3d(depth_left_wrist_float2, depth, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
			#depth_transformed_left_wrist_3d = [depth_left_wrist_float3.xyz.x, depth_left_wrist_float3.xyz.y, depth_left_wrist_float3.xyz.z]
   
			depth_spine_chest_float2 = pykinect.k4a_float2_t(depth_spine_chest_2d)
			depth = depth_image[int(depth_spine_chest_2d[1]), int(depth_spine_chest_2d[0])]
			depth_spine_chest_float3 = device.calibration.convert_2d_to_3d(depth_spine_chest_float2, depth, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
			depth_transformed_spine_chest_3d = [depth_spine_chest_float3.xyz.x, depth_spine_chest_float3.xyz.y, depth_spine_chest_float3.xyz.z]
	
			depth_spine_navel_float2 = pykinect.k4a_float2_t(depth_spine_navel_2d)
			depth = depth_image[int(depth_spine_navel_2d[1]), int(depth_spine_navel_2d[0])]
			depth_spine_navel_float3 = device.calibration.convert_2d_to_3d(depth_spine_navel_float2, depth, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
			depth_transformed_spine_navel_3d = [depth_spine_navel_float3.xyz.x, depth_spine_navel_float3.xyz.y, depth_spine_navel_float3.xyz.z]
   
			color_neck_3d = transformed_points_map[int(color_neck_2d[1]), int(color_neck_2d[0]), :]
			depth_neck_3d = points_map[int(depth_neck_2d[1]), int(depth_neck_2d[0]), :]
			neck_3d = skeleton_3d[pykinect.K4ABT_JOINT_NECK,:3]
			

			color_left_clavicle_3d = transformed_points_map[int(color_left_clavicle_2d[1]), int(color_left_clavicle_2d[0]), :]
			depth_left_clavicle_3d = points_map[int(depth_left_clavicle_2d[1]), int(depth_left_clavicle_2d[0]), :]
			left_clavicle_3d = skeleton_3d[pykinect.K4ABT_JOINT_CLAVICLE_LEFT,:3]
			
  		
			color_left_shoulder_3d = transformed_points_map[int(color_left_shoulder_2d[1]), int(color_left_shoulder_2d[0]), :]
			depth_left_shoulder_3d = points_map[int(depth_left_shoulder_2d[1]), int(depth_left_shoulder_2d[0]), :]
			left_shoulder_3d = skeleton_3d[pykinect.K4ABT_JOINT_SHOULDER_LEFT,:3]
			
    	
			#color_left_elbow_3d = transformed_points_map[int(color_left_elbow_2d[1]), int(color_left_elbow_2d[0]), :]
			#depth_left_elbow_3d = points_map[int(depth_left_elbow_2d[1]), int(depth_left_elbow_2d[0]), :]
			#left_elbow_3d = skeleton_3d[pykinect.K4ABT_JOINT_ELBOW_LEFT,:3]
			
   
			#color_left_wrist_3d = transformed_points_map[int(color_left_wrist_2d[1]), int(color_left_wrist_2d[0]), :]
			#depth_left_wrist_3d = points_map[int(depth_left_wrist_2d[1]), int(depth_left_wrist_2d[0]), :]
			#left_wrist_3d = skeleton_3d[pykinect.K4ABT_JOINT_WRIST_LEFT,:3]
			
   
			color_spine_chest_3d = transformed_points_map[int(color_spine_chest_2d[1]), int(color_spine_chest_2d[0]), :]
			depth_spine_chest_3d = points_map[int(depth_spine_chest_2d[1]), int(depth_spine_chest_2d[0]), :]
			spine_chest_3d = skeleton_3d[pykinect.K4ABT_JOINT_SPINE_CHEST,:3]
   
			color_spine_navel_3d = transformed_points_map[int(color_spine_navel_2d[1]), int(color_spine_navel_2d[0]), :]
			depth_spine_navel_3d = points_map[int(depth_spine_navel_2d[1]), int(depth_spine_navel_2d[0]), :]
			spine_navel_3d = skeleton_3d[pykinect.K4ABT_JOINT_SPINE_NAVEL,:3]
   
			#right
			color_right_clavicle_2d = color_skeleton_2d[pykinect.K4ABT_JOINT_CLAVICLE_RIGHT,:]
			depth_right_clavicle_2d = depth_skeleton_2d[pykinect.K4ABT_JOINT_CLAVICLE_RIGHT,:]

			color_right_shoulder_2d = color_skeleton_2d[pykinect.K4ABT_JOINT_SHOULDER_RIGHT,:]
			depth_right_shoulder_2d = depth_skeleton_2d[pykinect.K4ABT_JOINT_SHOULDER_RIGHT,:]
   
			color_right_elbow_2d = color_skeleton_2d[pykinect.K4ABT_JOINT_ELBOW_RIGHT,:]
			depth_right_elbow_2d = depth_skeleton_2d[pykinect.K4ABT_JOINT_ELBOW_RIGHT,:]
   
			color_right_wrist_2d = color_skeleton_2d[pykinect.K4ABT_JOINT_WRIST_RIGHT,:]
			depth_right_wrist_2d = depth_skeleton_2d[pykinect.K4ABT_JOINT_WRIST_RIGHT,:]
   
			#color_right_hip_2d = color_skeleton_2d[pykinect.K4ABT_JOINT_HIP_RIGHT,:]
			#depth_right_hip_2d = depth_skeleton_2d[pykinect.K4ABT_JOINT_HIP_RIGHT,:]
   
   
   
			depth_right_clavicle_float2 = pykinect.k4a_float2_t(depth_right_clavicle_2d)
			depth = depth_image[int(depth_right_clavicle_2d[1]), int(depth_right_clavicle_2d[0])]
			depth_right_clavicle_float3 = device.calibration.convert_2d_to_3d(depth_right_clavicle_float2, depth, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
			depth_transformed_right_clavicle_3d = [depth_right_clavicle_float3.xyz.x, depth_right_clavicle_float3.xyz.y, depth_right_clavicle_float3.xyz.z]
   
			depth_right_shoulder_float2 = pykinect.k4a_float2_t(depth_right_shoulder_2d)
			depth = depth_image[int(depth_right_shoulder_2d[1]), int(depth_right_shoulder_2d[0])]
			depth_right_shoulder_float3 = device.calibration.convert_2d_to_3d(depth_right_shoulder_float2, depth, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
			depth_transformed_right_shoulder_3d = [depth_right_shoulder_float3.xyz.x, depth_right_shoulder_float3.xyz.y, depth_right_shoulder_float3.xyz.z]
   
			depth_right_elbow_float2 = pykinect.k4a_float2_t(depth_right_elbow_2d)
			depth = depth_image[int(depth_right_elbow_2d[1]), int(depth_right_elbow_2d[0])]
			depth_right_elbow_float3 = device.calibration.convert_2d_to_3d(depth_right_elbow_float2, depth, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
			depth_transformed_right_elbow_3d = [depth_right_elbow_float3.xyz.x, depth_right_elbow_float3.xyz.y, depth_right_elbow_float3.xyz.z]
   
			depth_right_wrist_float2 = pykinect.k4a_float2_t(depth_right_wrist_2d)
			depth = depth_image[int(depth_right_wrist_2d[1]), int(depth_right_wrist_2d[0])]
			depth_right_wrist_float3 = device.calibration.convert_2d_to_3d(depth_right_wrist_float2, depth, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
			depth_transformed_right_wrist_3d = [depth_right_wrist_float3.xyz.x, depth_right_wrist_float3.xyz.y, depth_right_wrist_float3.xyz.z]
   
			#depth_right_hip_float2 = pykinect.k4a_float2_t(depth_right_hip_2d)
			#depth = depth_image[int(depth_right_hip_2d[1]), int(depth_right_hip_2d[0])]
			#depth_right_hip_float3 = device.calibration.convert_2d_to_3d(depth_right_hip_float2, depth, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
			#depth_transformed_right_hip_3d = [depth_right_hip_float3.xyz.x, depth_right_hip_float3.xyz.y, depth_right_hip_float3.xyz.z]

   
			color_right_clavicle_3d = transformed_points_map[int(color_right_clavicle_2d[1]), int(color_right_clavicle_2d[0]), :]
			depth_right_clavicle_3d = points_map[int(depth_right_clavicle_2d[1]), int(depth_right_clavicle_2d[0]), :]
			right_clavicle_3d = skeleton_3d[pykinect.K4ABT_JOINT_CLAVICLE_RIGHT,:3]
			
  		
			color_right_shoulder_3d = transformed_points_map[int(color_right_shoulder_2d[1]), int(color_right_shoulder_2d[0]), :]
			depth_right_shoulder_3d = points_map[int(depth_right_shoulder_2d[1]), int(depth_right_shoulder_2d[0]), :]
			right_shoulder_3d = skeleton_3d[pykinect.K4ABT_JOINT_SHOULDER_RIGHT,:3]
			
    	
			color_right_elbow_3d = transformed_points_map[int(color_right_elbow_2d[1]), int(color_right_elbow_2d[0]), :]
			depth_right_elbow_3d = points_map[int(depth_right_elbow_2d[1]), int(depth_right_elbow_2d[0]), :]
			right_elbow_3d = skeleton_3d[pykinect.K4ABT_JOINT_ELBOW_RIGHT,:3]
			
   
			color_right_wrist_3d = transformed_points_map[int(color_right_wrist_2d[1]), int(color_right_wrist_2d[0]), :]
			depth_right_wrist_3d = points_map[int(depth_right_wrist_2d[1]), int(depth_right_wrist_2d[0]), :]
			right_wrist_3d = skeleton_3d[pykinect.K4ABT_JOINT_WRIST_RIGHT,:3]
   
			#color_right_hip_3d = transformed_points_map[int(color_right_hip_2d[1]), int(color_right_hip_2d[0]), :]
			#depth_right_hip_3d = points_map[int(depth_right_hip_2d[1]), int(depth_right_hip_2d[0]), :]
			#right_hip_3d = skeleton_3d[pykinect.K4ABT_JOINT_HIP_RIGHT,:3]
			
			#elbows
			# Calculate vectors between the joints for elbows
			#vector_elbow_to_wrist_left = left_wrist_3d - left_elbow_3d
			#vector_shoulder_to_elbow_left = left_shoulder_3d - left_elbow_3d

			# Calculate the dot product and magnitude of the vectors
			#dot_product = np.dot(vector_elbow_to_wrist_left, vector_shoulder_to_elbow_left)
			#magnitude_elbow_to_wrist_left = np.linalg.norm(vector_elbow_to_wrist_left)
			#magnitude_shoulder_to_elbow_left = np.linalg.norm(vector_shoulder_to_elbow_left)

			# Calculate the angle between the vectors in radians
			#angle_radians_left_elbow = np.arccos(dot_product / (magnitude_elbow_to_wrist_left * magnitude_shoulder_to_elbow_left))
   
			#radians to degrees
			#angle_degrees_left_elbow = np.degrees(angle_radians_left_elbow)
   
			# Calculate vectors between the joints for right side
			vector_elbow_to_wrist_right = right_wrist_3d - right_elbow_3d
			vector_elbow_to_shoulder_right = right_shoulder_3d - right_elbow_3d
   

			vector_shoulder_to_elbow_right = depth_right_elbow_3d -depth_right_shoulder_3d
			vector_right_shoulder_to_left_shoulder = -depth_left_shoulder_3d + depth_right_shoulder_3d
   
			
			# Calculate the dot product and magnitude of the vectors for right side
			dot_product = np.dot(vector_elbow_to_wrist_right, vector_elbow_to_shoulder_right)
			magnitude_elbow_to_wrist_right = np.linalg.norm(vector_elbow_to_wrist_right)
			magnitude_shoulder_to_elbow_right = np.linalg.norm(vector_elbow_to_shoulder_right)


			# Calculate the angle between the vectors in radians for right side
			angle_radians_right_elbow = np.arccos(dot_product / (magnitude_elbow_to_wrist_right * magnitude_shoulder_to_elbow_right))

			# Convert the angle to degrees for right side *IMP IMP*
			angle_degrees_right_elbow = np.degrees(angle_radians_right_elbow)
			
			#if not np.isnan(angle_degrees_right_elbow) and not np.isnan(angle_degrees_left_elbow):
				#angles_list.append({"Left elbow angle": angle_degrees_left_elbow, "Right elbow angle": angle_degrees_right_elbow})
    
			
			#finding relevant vectors for shoulder orientation
			vector_left_to_right_clavicle = depth_right_clavicle_3d - depth_left_clavicle_3d
			vector_left_to_right_shoulder = depth_right_shoulder_3d - depth_left_shoulder_3d
			vector_neck_to_spine_chest = depth_spine_chest_3d - depth_neck_3d
			vector_chest_to_navel =  depth_spine_navel_3d - depth_spine_chest_3d
		
			normalized_left_to_right_shoulder = vector_left_to_right_shoulder / np.linalg.norm(vector_left_to_right_shoulder)
			normalized_neck_to_spine_chest = vector_neck_to_spine_chest / np.linalg.norm(vector_neck_to_spine_chest)
			normalized_shoulder_to_elbow_right = vector_shoulder_to_elbow_right/np.linalg.norm(vector_shoulder_to_elbow_right)
			normalized_chest_to_navel = vector_chest_to_navel/np.linalg.norm(vector_chest_to_navel)
   
			#we first find the forward facing vector
			forward_facing = np.cross(vector_left_to_right_shoulder, vector_chest_to_navel)
			normalized_forward_facing = forward_facing/np.linalg.norm(forward_facing)
			'''
			vector_right_hip_to_shoulder = depth_right_shoulder_3d-depth_right_hip_3d
			#trying wrt Hip Frame of Reference
   
   
			elbow_torso_plane_normal = np.cross(-vector_right_hip_to_shoulder, vector_shoulder_to_elbow_right)
			elbow_torso_plane_unit = elbow_torso_plane_normal/np.linalg.norm(elbow_torso_plane_normal)
			
			vector_body_plane_normal = np.cross(-vector_left_to_right_shoulder, -vector_right_hip_to_shoulder)
			vector_body_plane_unit = vector_body_plane_normal/np.linalg.norm(vector_body_plane_normal)
		
			a = np.dot(elbow_torso_plane_unit, vector_body_plane_unit)
			
			theta_1 = np.degrees(a)
			theta_2 = np.degrees(np.dot(-vector_right_hip_to_shoulder,vector_shoulder_to_elbow_right)/(np.linalg.norm(vector_right_hip_to_shoulder)*np.linalg.norm(vector_shoulder_to_elbow_right)))
			
			print(theta_2)
			'''
   			# Get quaternions and convert them to transfomation matrices
   
			
			translate_nc = [spine_chest_3d[0]-spine_navel_3d[0],spine_chest_3d[1]-spine_navel_3d[1],spine_chest_3d[2]-spine_navel_3d[2]]
			translate_clch = [right_clavicle_3d[0]-spine_chest_3d[0],right_clavicle_3d[1]-spine_chest_3d[1],right_clavicle_3d[2]-spine_chest_3d[2]]
			translate_sc = [right_shoulder_3d[0]-right_clavicle_3d[0],right_shoulder_3d[1]-right_clavicle_3d[1],right_shoulder_3d[2]-right_clavicle_3d[2]]
			
			body = body_frame.get_body(0)
			joint_info_right_shoulder =body.joints[pykinect.K4ABT_JOINT_SHOULDER_RIGHT].orientation
			quat_right_shoulder_dict = joint_info_right_shoulder.__iter__()
			vector_4d_right_shoulder = np.array([quat_right_shoulder_dict['w'],quat_right_shoulder_dict['x'],quat_right_shoulder_dict['y'],quat_right_shoulder_dict['z']])
			quat_right_shoulder = pq(array = vector_4d_right_shoulder)
			unit_quat_right_shoulder = quat_right_shoulder.normalised
			t_shoulder = quat_right_shoulder.transformation_matrix
			

			joint_type_right_shoulder = pykinect.K4ABT_JOINT_SHOULDER_RIGHT
			unit_quat_right_shoulder, t_shoulder = get_unit_quat_and_transformation_matrix(body_frame, joint_type_right_shoulder)
			joint_type_right_clavicle = pykinect.K4ABT_JOINT_CLAVICLE_RIGHT
			unit_quat_right_clavicle, t_clavicle = get_unit_quat_and_transformation_matrix(body_frame, joint_type_right_clavicle)
			joint_type_chest = pykinect.K4ABT_JOINT_SPINE_CHEST
			unit_quat_chest, t_chest = get_unit_quat_and_transformation_matrix(body_frame, joint_type_chest)	
			joint_type_navel = pykinect.K4ABT_JOINT_SPINE_NAVEL
			unit_quat_navel, t_navel = get_unit_quat_and_transformation_matrix(body_frame, joint_type_navel)
			joint_type_pelvis = pykinect.K4ABT_JOINT_PELVIS
			unit_quat_pelvis, t_pelvis = get_unit_quat_and_transformation_matrix(body_frame, joint_type_pelvis)

			Q_shoulder_to_camera = pq(np.array([1, 0, 0, 0]))
			Q_shoulder_to_camera = (Q_shoulder_to_camera* unit_quat_right_shoulder)
			Q_shoulder_to_camera = (Q_shoulder_to_camera*unit_quat_right_clavicle)
			Q_shoulder_to_camera = (Q_shoulder_to_camera* unit_quat_chest)
			Q_shoulder_to_camera = (Q_shoulder_to_camera* unit_quat_navel)
			Q_shoulder_to_camera = (Q_shoulder_to_camera*unit_quat_pelvis)
			
			A = Q_shoulder_to_camera.rotation_matrix  


			



			#resultant_quaternion = unit_quat_right_shoulder * unit_quat_right_clavicle* unit_quat_chest * unit_quat_navel * unit_quat_pelvis
			#r_resultant = resultant_quaternion.rotation_matrix

			
			
			'''
			for i in range(3):
				t_shoulder[i][3] = translate_sc[i]
				t_clavicle[i][3] = translate_clch[i]
				t_chest[i][3] = translate_nc[i]
			r_is_x =[[1, 0, 0],
       			  [0, cos(-90), -sin(-90)],
                  [0, sin(-90), cos(-90)]]
			r_is_z = [[cos(-90), -sin(-90), 0],
       				 [sin(-90), cos(-90), 0],
          	  	 	 [0, 0, 1]]
			del_x = right_shoulder_3d[0] - spine_navel_3d[0]
			del_y = right_shoulder_3d[1] - spine_navel_3d[1]
			del_z = right_shoulder_3d[2] - spine_navel_3d[2]
			r_is = np.matmul(r_is_x,r_is_z)
			t_is = np.array([[r_is[0][0],r_is[0][1],r_is[0][2],del_x],
           			[r_is[1][0],r_is[1][1],r_is[1][2],del_y],
              		[r_is[2][0],r_is[2][1],r_is[2][2],del_z],
                	[0,0,0,1]])
		
			inv_t_is = np.linalg.inv(t_is)
			
			formatted_matrix =np.array(t_shoulder @ t_clavicle @ t_chest @ inv_t_is)
		
			pitch = np.degrees(np.arctan2(-formatted_matrix[0][2], np.sqrt(formatted_matrix[0][0]*formatted_matrix[0][0] + formatted_matrix[1][0]*formatted_matrix[1][0])))
			yaw = np.degrees(np.arctan2(formatted_matrix[0][1], formatted_matrix[0][0]))
			roll = np.degrees(np.arctan2(formatted_matrix[1][0], formatted_matrix[1][1]))
			print(roll,pitch,yaw)
			'''
   			#squat_right_shoulder = pq(axis=[quat_right_shoulder_dict['x'],quat_right_shoulder_dict['y'],quat_right_shoulder_dict['z']], angle=quat_right_shoulder_dict['w'])
			
			'''
			joint_info_right_clavicle =body.joints[pykinect.K4ABT_JOINT_CLAVICLE_RIGHT].orientation
			quat_right_clavicle_dict = joint_info_right_clavicle.__iter__()
			quat_right_clavicle = pq(axis=[quat_right_clavicle_dict['x'],quat_right_clavicle_dict['y'],quat_right_clavicle_dict['z']], angle=quat_right_clavicle_dict['w'])
			v_prime_right_clavicle = quat_right_clavicle.rotate(np.array([1,0,0]))
			
			
			joint_info_right_clavicle =body.joints[pykinect.K4ABT_JOINT_CLAVICLE_RIGHT].orientation
			quat_right_clavicle_dict = joint_info_right_shoulder.__iter__()
			vector_4d_right_clavicle = np.array([quat_right_clavicle_dict['w'],quat_right_clavicle_dict['x'],quat_right_clavicle_dict['y'],quat_right_clavicle_dict['z']])
			quat_right_clavicle = pq(array = vector_4d_right_clavicle)
			unit_quat_right_clavicle = quat_right_clavicle.normalised
			t_clavicle = quat_right_clavicle.transformation_matrix
			
			
			joint_info_chest =body.joints[pykinect.K4ABT_JOINT_SPINE_CHEST].orientation
			quat_chest_dict = joint_info_chest.__iter__()
			vector_4d_chest= np.array([quat_chest_dict['w'],quat_chest_dict['x'],quat_chest_dict['y'],quat_chest_dict['z']])
			quat_chest = pq(array = vector_4d_chest)
			unit_quat_chest = quat_chest.normalised
			t_chest = quat_chest.transformation_matrix
			
   
			joint_info_navel =body.joints[pykinect.K4ABT_JOINT_SPINE_NAVEL].orientation
			quat_navel_dict = joint_info_navel.__iter__()
			vector_4d_navel= np.array([quat_navel_dict['w'],quat_navel_dict['x'],quat_navel_dict['y'],quat_navel_dict['z']])
			quat_navel = pq(array = vector_4d_navel)
			unit_quat_navel = quat_navel.normalised
			
			rotation_chest = R.from_quat(vector_4d_chest)
			rot_matrix_chest = rotation_chest.as_matrix()
			x_axis_chest = np.array(rot_matrix_chest[:, 0])
			y_axis_chest = np.array(rot_matrix_chest[:, 1])
			z_axis_chest = np.array(rot_matrix_chest[:, 2])	
			
			
			rotation_right_clavicle = R.from_quat(vector_4d_right_clavicle)
			rot_matrix_right_clavicle = rotation_right_clavicle.as_matrix()
			x_axis_right_clavicle= np.array(rot_matrix_right_clavicle[:, 0])
			y_axis_right_clavicle = np.array(rot_matrix_right_clavicle[:, 1])
			z_axis_right_clavicle = np.array(rot_matrix_right_clavicle[:, 2])
   
			rotation_right_shoulder = R.from_quat(vector_4d_right_shoulder)
			rot_matrix_right_shoulder = rotation_right_shoulder.as_matrix()
			x_axis_right_shoulder= np.array(rot_matrix_right_shoulder[:, 0])
			y_axis_right_shoulder = np.array(rot_matrix_right_shoulder[:, 1])
			z_axis_right_shoulder = np.array(rot_matrix_right_shoulder[:, 2])
			'''

			
			
			""" #ANGle between cam plane and body plane
			dot_product_planes = np.dot([0,0,1],normalized_forward_facing)
			angle_radians_planes = np.arccos(dot_product_planes)
			angle_degrees_planes = np.degrees(angle_radians_planes) """
			
		
			#z
   
			# Project the "Shoulder to Elbow" Vector onto the Plane
			dot_product_z = np.dot(normalized_shoulder_to_elbow_right, normalized_forward_facing)
			z_projection = normalized_shoulder_to_elbow_right - dot_product_z * normalized_forward_facing
			angle_radians_z = np.arccos(np.dot(z_projection, vector_left_to_right_shoulder) / (np.linalg.norm(z_projection) * np.linalg.norm(vector_left_to_right_shoulder)))
	
			angle_degrees_z = np.degrees(angle_radians_z)
			list_degrees_z_list.append(angle_degrees_z )
			if len(list_degrees_z_list) >= max_values:
				average_angle_z = np.mean(list_degrees_z_list)	
				list_degrees_z_list =[] 
			#x
			
			
			dot_product_x = np.dot(normalized_left_to_right_shoulder,normalized_shoulder_to_elbow_right)
			x_projection = normalized_shoulder_to_elbow_right - dot_product_x*normalized_left_to_right_shoulder

			angle_radians_x = np.arccos(np.dot(x_projection,normalized_chest_to_navel)/(np.linalg.norm(x_projection) * np.linalg.norm(normalized_chest_to_navel)))
			angle_degrees_x =np.degrees(angle_radians_x)
			list_degrees_x_list.append(angle_degrees_x)
			if len(list_degrees_x_list) >= max_values:
				average_angle_x = np.mean(list_degrees_x_list)
				list_degrees_x_list =[]
    		#y
      
			
			dot_product_y = np.dot(normalized_shoulder_to_elbow_right,normalized_chest_to_navel)
			y_projection = normalized_shoulder_to_elbow_right - dot_product_y*normalized_chest_to_navel
			a = np.dot(y_projection,vector_left_to_right_shoulder)
			b = np.linalg.norm(y_projection)
			c = np.linalg.norm(vector_left_to_right_shoulder) 

			angle_radians_y = np.arccos(a/(b * c))

			angle_degrees_y =np.degrees(angle_radians_y)
			list_degrees_y_list.append(angle_degrees_y)
			if len(list_degrees_y_list) >= max_values:
				average_angle_y = np.mean(list_degrees_y_list)
				list_degrees_y_list =[] 
			np.set_printoptions(precision=4, suppress=True)
			A,B,C= angle_degrees_z, angle_degrees_y, angle_degrees_x
			
			
			
		
        # Draw the skeletons into the color image
		color_skeleton = body_frame.draw_bodies(color_image, pykinect.K4A_CALIBRATION_TYPE_COLOR)
		transformed_color_skeleton = body_frame.draw_bodies(transformed_color_image, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
		
		# Overlay body segmentation on depth image
		#position_left = (int(color_left_elbow_2d[0]), int(color_left_elbow_2d[1])) 
		position_right_elbow = (int(color_right_elbow_2d[0]), int(color_right_elbow_2d[1])) 
		

		

		x = color_skeleton.shape[1] -1850
		y = color_skeleton.shape[0] - 1000
		#text = f'alpha : {angle_x_deg:.2f} beta : {angle_y_deg:.2f} gamma :{angle_z_deg:.2f} Iteration : {iteration}'
		#cv2.putText(color_skeleton, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,2, (255, 0, 255), 2)

		cv2.imshow('Color image with skeleton', color_skeleton)

     
		
		if cv2.waitKey(1) == ord('q'):
			break
			break