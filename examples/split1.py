import cv2
import numpy as np
import pandas as pd
import pykinect_azure as pykinect
from scipy.spatial.transform import Rotation as R
import split2 as sp

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

	#cv2.namedWindow('Color image with skeleton',cv2.WINDOW_NORMAL)
	#cv2.namedWindow('Transformed Color image with skeleton',cv2.WINDOW_NORMAL)
while True:
	body_frame, color_image, transformed_color_image, color_right_elbow_2d,angle_degrees_right_elbow	 = sp.acquire_and_calculate_angles(device, bodyTracker)
	 # Draw the skeletons into the color image
	color_skeleton = body_frame.draw_bodies(color_image, pykinect.K4A_CALIBRATION_TYPE_COLOR)
	transformed_color_skeleton = body_frame.draw_bodies(transformed_color_image, pykinect.K4A_CALIBRATION_TYPE_DEPTH)

		# Overlay body segmentation on depth image
	position_right = (int(color_right_elbow_2d[0]), int(color_right_elbow_2d[1])) 
 
	cv2.putText(color_skeleton,f'Alpha: {angle_degrees_right_elbow:.2f}', position_right, cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 2)
	cv2.imshow('Color image with skeleton', color_skeleton)
	cv2.imshow('Transformed Color image with skeleton', transformed_color_skeleton)
     
	if cv2.waitKey(1) == ord('q'):
			break


device.stop()
bodyTracker.stop()
cv2.destroyAllWindows()
 