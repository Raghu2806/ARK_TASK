import cv2
import numpy as np

# Load the left and right images
left_image = cv2.imread('left.png')
right_image = cv2.imread('right.png')

# Convert images to grayscale
left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

# Set up the StereoSGBM object
window_size = 5
min_disp = 0
num_disp = 64
stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=window_size)

# Compute the disparity map
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

# Normalize the disparity map
disparity_norm = cv2.normalize(disparity, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# Create a colormap
colormap = cv2.applyColorMap(np.uint8(disparity_norm * 255), cv2.COLORMAP_JET)

# Save the colormap as an image
cv2.imwrite('depth.png', colormap)