import cv2
import numpy as np

# Load the image
img = cv2.imread("table.png")

# Resize the image
resized_img = cv2.resize(img, (600, 518), interpolation=cv2.INTER_AREA)

# Saving the resized image
cv2.imwrite("resized_img.png", resized_img)

# Convert the image to grayscale
gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

# Saving the grayscale image
cv2.imwrite("gray.png", gray)

# Applying blur for Sobel
image_blur = cv2.medianBlur(gray, 9)

# Get the image dimensions
height, width = image_blur.shape

# Define the Sobel kernels
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Initialize the output image
edges = np.zeros_like(image_blur)

# Perform the Sobel operator
for y in range(1, height - 1):
    for x in range(1, width - 1):
        gx = np.sum(image_blur[y - 1:y + 2, x - 1:x + 2] * sobel_x)
        gy = np.sum(image_blur[y - 1:y + 2, x - 1:x + 2] * sobel_y)
        edges[y, x] = np.sqrt(gx**2 + gy**2)

# Save the edge image
cv2.imwrite("edge.png", edges)

# Binarize the image
_, binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Save the binarized image
cv2.imwrite("binary.png", binary)

# Detect lines using Hough transform on the binary image
lines = cv2.HoughLines(binary, 1, np.pi / 180, 200)

# Create a new color image for drawing lines (consider using the original image size)
line_img = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)  # Copy to avoid modifying original

# Draw the lines on the new image (change color here)
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw red lines (BGR format)

# Save the image with lines
cv2.imwrite("hough_lines_binary.png", line_img)

# Display the image with lines
cv2.imshow('Hough Lines on Binary Image', line_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
