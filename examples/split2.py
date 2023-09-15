import cv2
import numpy as np
import pandas as pd
import pykinect_azure as pykinect


from scipy.spatial.transform import Rotation as R

def acquire_and_calculate_angles(device, bodyTracker):		
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

	#if not ret_color or not ret_depth or not ret_point or not ret_transformed_point or not ret_transformed_color:


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
   
		color_left_elbow_2d = color_skeleton_2d[pykinect.K4ABT_JOINT_ELBOW_LEFT,:]
		depth_left_elbow_2d = depth_skeleton_2d[pykinect.K4ABT_JOINT_ELBOW_LEFT,:]
   
		color_left_wrist_2d = color_skeleton_2d[pykinect.K4ABT_JOINT_WRIST_LEFT,:]
		depth_left_wrist_2d = depth_skeleton_2d[pykinect.K4ABT_JOINT_WRIST_LEFT,:]
   
		color_spine_chest_2d = color_skeleton_2d[pykinect.K4ABT_JOINT_SPINE_CHEST,:]
		depth_spine_chest_2d = depth_skeleton_2d[pykinect.K4ABT_JOINT_SPINE_CHEST,:]


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
   
		depth_left_elbow_float2 = pykinect.k4a_float2_t(depth_left_elbow_2d)
		depth = depth_image[int(depth_left_elbow_2d[1]), int(depth_left_elbow_2d[0])]
		depth_left_elbow_float3 = device.calibration.convert_2d_to_3d(depth_left_elbow_float2, depth, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
		depth_transformed_left_elbow_3d = [depth_left_elbow_float3.xyz.x, depth_left_elbow_float3.xyz.y, depth_left_elbow_float3.xyz.z]
   
		depth_left_wrist_float2 = pykinect.k4a_float2_t(depth_left_wrist_2d)
		depth = depth_image[int(depth_left_wrist_2d[1]), int(depth_left_wrist_2d[0])]
		depth_left_wrist_float3 = device.calibration.convert_2d_to_3d(depth_left_wrist_float2, depth, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
		depth_transformed_left_wrist_3d = [depth_left_wrist_float3.xyz.x, depth_left_wrist_float3.xyz.y, depth_left_wrist_float3.xyz.z]
   
		depth_spine_chest_float2 = pykinect.k4a_float2_t(depth_spine_chest_2d)
		depth = depth_image[int(depth_spine_chest_2d[1]), int(depth_spine_chest_2d[0])]
		depth_spine_chest_float3 = device.calibration.convert_2d_to_3d(depth_spine_chest_float2, depth, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
		depth_transformed_spine_chest_3d = [depth_spine_chest_float3.xyz.x, depth_spine_chest_float3.xyz.y, depth_spine_chest_float3.xyz.z]
	
   
		color_neck_3d = transformed_points_map[int(color_neck_2d[1]), int(color_neck_2d[0]), :]
		depth_neck_3d = points_map[int(depth_neck_2d[1]), int(depth_neck_2d[0]), :]
		neck_3d = skeleton_3d[pykinect.K4ABT_JOINT_NECK,:3]
			

		color_left_clavicle_3d = transformed_points_map[int(color_left_clavicle_2d[1]), int(color_left_clavicle_2d[0]), :]
		depth_left_clavicle_3d = points_map[int(depth_left_clavicle_2d[1]), int(depth_left_clavicle_2d[0]), :]
		left_clavicle_3d = skeleton_3d[pykinect.K4ABT_JOINT_CLAVICLE_LEFT,:3]
			
  		
		color_left_shoulder_3d = transformed_points_map[int(color_left_shoulder_2d[1]), int(color_left_shoulder_2d[0]), :]
		depth_left_shoulder_3d = points_map[int(depth_left_shoulder_2d[1]), int(depth_left_shoulder_2d[0]), :]
		left_shoulder_3d = skeleton_3d[pykinect.K4ABT_JOINT_SHOULDER_LEFT,:3]
			
    	
		color_left_elbow_3d = transformed_points_map[int(color_left_elbow_2d[1]), int(color_left_elbow_2d[0]), :]
		depth_left_elbow_3d = points_map[int(depth_left_elbow_2d[1]), int(depth_left_elbow_2d[0]), :]
		left_elbow_3d = skeleton_3d[pykinect.K4ABT_JOINT_ELBOW_LEFT,:3]
			
   
		color_left_wrist_3d = transformed_points_map[int(color_left_wrist_2d[1]), int(color_left_wrist_2d[0]), :]
		depth_left_wrist_3d = points_map[int(depth_left_wrist_2d[1]), int(depth_left_wrist_2d[0]), :]
		left_wrist_3d = skeleton_3d[pykinect.K4ABT_JOINT_WRIST_LEFT,:3]
			
   
		color_spine_chest_3d = transformed_points_map[int(color_spine_chest_2d[1]), int(color_spine_chest_2d[0]), :]
		depth_spine_chest_3d = points_map[int(depth_spine_chest_2d[1]), int(depth_spine_chest_2d[0]), :]
		spine_chest_3d = skeleton_3d[pykinect.K4ABT_JOINT_SPINE_CHEST,:3]
   
			#right
		color_right_clavicle_2d = color_skeleton_2d[pykinect.K4ABT_JOINT_CLAVICLE_RIGHT,:]
		depth_right_clavicle_2d = depth_skeleton_2d[pykinect.K4ABT_JOINT_CLAVICLE_RIGHT,:]

		color_right_shoulder_2d = color_skeleton_2d[pykinect.K4ABT_JOINT_SHOULDER_RIGHT,:]
		depth_right_shoulder_2d = depth_skeleton_2d[pykinect.K4ABT_JOINT_SHOULDER_RIGHT,:]
   
		color_right_elbow_2d = color_skeleton_2d[pykinect.K4ABT_JOINT_ELBOW_RIGHT,:]
		depth_right_elbow_2d = depth_skeleton_2d[pykinect.K4ABT_JOINT_ELBOW_RIGHT,:]
   
		color_right_wrist_2d = color_skeleton_2d[pykinect.K4ABT_JOINT_WRIST_RIGHT,:]
		depth_right_wrist_2d = depth_skeleton_2d[pykinect.K4ABT_JOINT_WRIST_RIGHT,:]
   
   
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
			
			#elbows
			# Calculate vectors between the joints for elbows
		vector_elbow_to_wrist_left = left_wrist_3d - left_elbow_3d
		vector_shoulder_to_elbow_left = left_shoulder_3d - left_elbow_3d

			# Calculate the dot product and magnitude of the vectors
		dot_product = np.dot(vector_elbow_to_wrist_left, vector_shoulder_to_elbow_left)
		magnitude_elbow_to_wrist_left = np.linalg.norm(vector_elbow_to_wrist_left)
		magnitude_shoulder_to_elbow_left = np.linalg.norm(vector_shoulder_to_elbow_left)

			# Calculate the angle between the vectors in radians
		angle_radians_left_elbow = np.arccos(dot_product / (magnitude_elbow_to_wrist_left * magnitude_shoulder_to_elbow_left))
   
			#radians to degrees
		angle_degrees_left_elbow = np.degrees(angle_radians_left_elbow)
   
			# Calculate vectors between the joints for right side
		vector_elbow_to_wrist_right = right_wrist_3d - right_elbow_3d
		vector_elbow_to_shoulder_right = right_shoulder_3d - right_elbow_3d
   

		vector_shoulder_to_elbow_right = np.array(right_elbow_3d) - np.array(right_shoulder_3d) 
		vector_shoulder_to_clavicle_right = np.array(right_clavicle_3d) - np.array(right_shoulder_3d) 

			# Calculate the dot product and magnitude of the vectors for right side
		dot_product = np.dot(vector_elbow_to_wrist_right, vector_elbow_to_shoulder_right)
		magnitude_elbow_to_wrist_right = np.linalg.norm(vector_elbow_to_wrist_right)
		magnitude_shoulder_to_elbow_right = np.linalg.norm(vector_elbow_to_shoulder_right)


			# Calculate the angle between the vectors in radians for right side
		angle_radians_right_elbow = np.arccos(dot_product / (magnitude_elbow_to_wrist_right * magnitude_shoulder_to_elbow_right))

			# Convert the angle to degrees for right side *IMP IMP*
		angle_degrees_right_elbow = np.degrees(angle_radians_right_elbow)
			
   
		if not np.isnan(angle_degrees_right_elbow) and not np.isnan(angle_degrees_left_elbow):
			angles_list = [0,0,0,angle_degrees_right_elbow,0,0,0]
   
	return body_frame, color_image, transformed_color_image, color_right_elbow_2d,angle_degrees_right_elbow	
    

	


 
if __name__ == "__main__":
    # Initialize the library, if the library is not found, add the library path as an argument
    pykinect.initialize_libraries(track_body=True)

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

    # Start device
    device = pykinect.start_device(config=device_config)

    # Start body tracker
    bodyTracker = pykinect.start_body_tracker()

    angles_list = acquire_and_calculate_angles(device, bodyTracker)

    # Release the camera
    device.stop()

    # Process the angles_list as needed (e.g., convert to a vector)
    angles_vector = np.array(angles_list)

    # Print or use the angles_vector as needed
    print("Angles Vector:", angles_vector)


    
