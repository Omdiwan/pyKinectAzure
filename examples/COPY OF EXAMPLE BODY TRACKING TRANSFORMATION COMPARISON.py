import cv2
import numpy as np
import pandas as pd
import pykinect_azure as pykinect
from openpyxl import Workbook
import csv
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":

	# Initialize the library, if the library is not found, add the library path as argument
	pykinect.initialize_libraries(track_body=True)

	# Modify camera configuration
	device_config = pykinect.default_configuration
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
	device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
	#print(device_config)

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
	fig = plt.figure()
	ax = fig.add_subplot(111,projection = '3d')
 
	

	plt.ion()
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
   

			vector_shoulder_to_elbow_right = right_elbow_3d - right_shoulder_3d
			vector_right_shoulder_to_left_shoulder = -left_shoulder_3d + right_shoulder_3d

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
			vector_left_to_right_clavicle = right_clavicle_3d - left_clavicle_3d
			vector_left_to_right_shoulder = right_shoulder_3d - left_shoulder_3d
			vector_neck_to_spine_chest = spine_chest_3d - neck_3d
			vector_chest_to_navel =  spine_navel_3d - spine_chest_3d
		
			normalized_left_to_right_shoulder = vector_left_to_right_shoulder / np.linalg.norm(vector_left_to_right_shoulder)
			normalized_neck_to_spine_chest = vector_neck_to_spine_chest / np.linalg.norm(vector_neck_to_spine_chest)
			normalized_shoulder_to_elbow_right = vector_shoulder_to_elbow_right/np.linalg.norm(vector_shoulder_to_elbow_right)
			normalized_chest_to_navel = vector_chest_to_navel/np.linalg.norm(vector_chest_to_navel)
			#we first find the forward facing vector
			forward_facing = np.cross(vector_left_to_right_shoulder, vector_chest_to_navel)
			normalized_forward_facing = forward_facing/np.linalg.norm(forward_facing)
			
			#ANGle between cam plane and body plane
			dot_product_planes = np.dot([0,0,1],normalized_forward_facing)
			angle_radians_planes = np.arccos(dot_product_planes)
			angle_degrees_planes = np.degrees(angle_radians_planes)
			
			
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
			a = np.dot(y_projection,normalized_forward_facing)
			b = np.linalg.norm(y_projection)
			c = np.linalg.norm(normalized_forward_facing) 

			angle_radians_y = np.arccos(a/(b * c))

			angle_degrees_y =np.degrees(angle_radians_y)
			list_degrees_y_list.append(angle_degrees_y)
			if len(list_degrees_y_list) >= max_values:
				average_angle_y = np.mean(list_degrees_y_list)
				list_degrees_y_list =[]
			
			
			#print(np.degrees(np.arccos(np.dot(vector_chest_to_navel,vector_left_to_right_shoulder)/(np.linalg.norm(vector_chest_to_navel)*np.linalg.norm(vector_left_to_right_shoulder)))))
			
			print(	iteration,
         			np.round(180 - angle_degrees_planes,2),
					np.round(angle_degrees_y,2),
					np.round(a,2),
					np.round(b,2),
					np.round(c,2),
					np.round(y_projection,2),
					np.round(dot_product_y,2),
					np.round(vector_shoulder_to_elbow_right,2),
         			np.round(right_shoulder_3d,2), 
            		np.round(right_elbow_3d,2),
					np.round(vector_left_to_right_shoulder,2),
              		np.round(left_shoulder_3d,2),
                	np.round(right_shoulder_3d,2),
					np.round(vector_chest_to_navel,2),
                	np.round(spine_chest_3d,2),
                 	np.round(spine_navel_3d,2)
                )
			#print(iteration, np.round(angle_degrees_y, 2), np.round(a, 2), np.round(b, 2), np.round(c, 2), np.round(y_projection, 2), np.round(vector_left_to_right_shoulder, 2), np.round(dot_product_y, 2), np.round(normalized_neck_to_spine_chest, 4), np.round(vector_shoulder_to_elbow_right, 2), np.round(right_elbow_3d, 2), np.round(right_shoulder_3d, 2), np.round(vector_neck_to_spine_chest, 2), np.round(spine_chest_3d, 2), np.round(neck_3d, 2))
		
		#visualization of vectors
		ax.clear()
		for art in ax.artists:
			art.remove()
			
		
		ax.quiver(100, 100, 100, vector_left_to_right_shoulder[0], vector_left_to_right_shoulder[1], vector_left_to_right_shoulder[2], color='g', label='Left to Right Shoulder Vector')
		ax.quiver(100, 100, 100, vector_chest_to_navel[0], vector_chest_to_navel[1], vector_chest_to_navel[2], color='r', label='chest to navel ')
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		ax.set_title('Relevant Vectors Visualization')
		ax.legend()
		plt.show()
 
    
    
        # Draw the skeletons into the color image
		color_skeleton = body_frame.draw_bodies(color_image, pykinect.K4A_CALIBRATION_TYPE_COLOR)
		transformed_color_skeleton = body_frame.draw_bodies(transformed_color_image, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
		
		# Overlay body segmentation on depth image
		#position_left = (int(color_left_elbow_2d[0]), int(color_left_elbow_2d[1])) 
		position_right_elbow = (int(color_right_elbow_2d[0]), int(color_right_elbow_2d[1])) 
		text = f"X Angle: {angle_degrees_x:.2f} Y Angle: {angle_degrees_y:.2f} Z Angle: {angle_degrees_z:.2f} Iteration: {iteration}"



		x = color_skeleton.shape[1] -1850
		y = 50
		cv2.putText(color_skeleton,text , (x, y), cv2.FONT_HERSHEY_DUPLEX, 1.75, (255, 0, 255), 2)
		cv2.imshow('Color image with skeleton', color_skeleton)

		plt.draw()
		plt.pause(0.01)

     
		
		if cv2.waitKey(1) == ord('q'):
			break
 