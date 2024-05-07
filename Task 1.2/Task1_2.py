import cv2
import numpy as np

# Load the left and right images
left_image = cv2.imread('left.png')
right_image = cv2.imread('right.png')

# Convert images to grayscale
left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

# Define parameters
window_size = 5
max_disp = 64

# Compute the disparity map using block matching
height, width = left_gray.shape
disparity_map = np.zeros((height, width), dtype=np.float32)

for y in range(height):
    for x in range(max_disp, width):
        min_cost = float('inf')
        best_disp = 0
        for disp in range(max_disp):
            left_start_y = max(0, y - window_size // 2)
            left_end_y = min(height, y + window_size // 2 + 1)
            left_start_x = max(0, x - window_size // 2)
            left_end_x = min(width, x + window_size // 2 + 1)
            right_start_y = max(0, y - window_size // 2)
            right_end_y = min(height, y + window_size // 2 + 1)
            right_start_x = max(0, x - disp - window_size // 2)
            right_end_x = min(width, x - disp + window_size // 2 + 1)
            left_window = left_gray[left_start_y:left_end_y, left_start_x:left_end_x]
            right_window = right_gray[right_start_y:right_end_y, right_start_x:right_end_x]

            # Check if both windows have the same shape
            if left_window.shape == right_window.shape:
              
                cost = np.sum(np.abs(left_window.astype(np.int32) - right_window.astype(np.int32)))

                if cost < min_cost:
                    min_cost = cost
                    best_disp = disp

        # Store the best disparity in the disparity map
        disparity_map[y, x] = best_disp

# Normalize the disparity map
disparity_norm = cv2.normalize(disparity_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# Create a colormap
colormap = cv2.applyColorMap(np.uint8(disparity_norm * 255), cv2.COLORMAP_JET)

# Save the colormap as an image
cv2.imwrite('depth1.png', colormap)
