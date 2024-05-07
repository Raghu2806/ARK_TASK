import cv2
import numpy as np
import random
from collections import deque

# Load the maze image
img = cv2.imread('maze.png')

# Define start and end points for easy scenario
hard_start_point = (150, 10)
hard_end_point = (420,290)

# Define start and end points for hard scenario
easy_start_point = (10, 300)
easy_end_point = (85, 300)

# Define obstacle space
obstacle_space = np.where(img[:, :, 0] == 0)

# Function to check if a point is in an obstacle
def is_obstacle(point):
    return img[point[1], point[0], 0] == 0

# Function to generate random points
def generate_random_point():
    x = random.randint(0, img.shape[1] - 1)
    y = random.randint(0, img.shape[0] - 1)
    return (x, y)

# Function to check if a line intersects an obstacle
def line_intersects_obstacle(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    points = np.linspace(p1, p2, num=int(np.hypot(x2 - x1, y2 - y1)))
    for point in points:
        if is_obstacle((int(point[0]), int(point[1]))):
            return True
    return False

# PRM algorithm
def prm_algorithm(start, end):
    roadmap = []
    n_points = 500
    for i in range(n_points):
        point = generate_random_point()
        if not is_obstacle(point):
            roadmap.append(point)

    # Add start and end points to the roadmap
    roadmap.append(start)
    roadmap.append(end)

    # Connect neighboring points
    adjacency_list = {node: [] for node in roadmap}
    for i in range(len(roadmap)):
        for j in range(i + 1, len(roadmap)):
            p1 = roadmap[i]
            p2 = roadmap[j]
            if not line_intersects_obstacle(p1, p2):
                adjacency_list[p1].append(p2)
                adjacency_list[p2].append(p1)

    # Find path using BFS
    queue = deque([(start, [start])])
    visited = set()
    while queue:
        current, path = queue.popleft()
        if current == end:
            return path
        visited.add(current)
        for neighbor in adjacency_list[current]:
            if neighbor not in visited:
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))
    return None

# Find path for the easy scenario
easy_path = prm_algorithm(easy_start_point, easy_end_point)

# Find path for the hard scenario
hard_path = prm_algorithm(hard_start_point, hard_end_point)

# Save the images
if easy_path:
    easy_img = img.copy()
    for i in range(len(easy_path) - 1):
        p1 = easy_path[i]
        p2 = easy_path[i + 1]
        cv2.line(easy_img, p1, p2, (0, 0, 255), 2)
    cv2.imwrite('easy_path.png', easy_img)
else:
    print("No easy path found!")

if hard_path:
    hard_img = img.copy()
    for i in range(len(hard_path) - 1):
        p1 = hard_path[i]
        p2 = hard_path[i + 1]
        cv2.line(hard_img, p1, p2, (0, 0, 255), 2)
    cv2.imwrite('hard_path.png', hard_img)
else:
    print("No hard path found!")
