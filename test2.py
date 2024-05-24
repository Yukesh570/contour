import cv2
import numpy as np
import itertools
import matplotlib.pyplot as plt

def detect_largest_triangle(image_path, min_triangle_area=1000):
    # Load the image
    image = cv2.imread(r'C:\Users\Yukesh\Downloads\snookervideo\balls.jpg')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=50, param2=30, minRadius=20,
                               maxRadius=100)

    # If circles are detected
    if circles is not None:
        circles = np.uint16(np.around(circles))
        centers = circles[0, :, :2]

        # Initialize variables to store the largest triangle found
        max_area = 0
        max_triangle = None

        # Generate combinations of circle centers to form triangles
        combinations = list(itertools.combinations(centers, 3))

        # Check if each combination forms a triangle
        for combination in combinations:
            # Calculate area of the triangle using cross product
            area = 0.5 * abs(np.cross(combination[1] - combination[0], combination[2] - combination[0]))

            # Update the largest triangle if the current triangle has a greater area
            if area > max_area and area > min_triangle_area:
                max_area = area
                max_triangle = combination

        # If a triangle with a valid area is found, draw it on the image
        if max_triangle is not None:
            cv2.line(image, tuple(max_triangle[0]), tuple(max_triangle[1]), (0, 255, 0), 2)
            cv2.line(image, tuple(max_triangle[1]), tuple(max_triangle[2]), (0, 255, 0), 2)
            cv2.line(image, tuple(max_triangle[2]), tuple(max_triangle[0]), (0, 255, 0), 2)

    # Convert BGR to RGB for displaying with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

# Example usage:
image_path = 'balls_image.jpg'  # Replace 'balls_image.jpg' with the path to your image
detect_largest_triangle(image_path)
