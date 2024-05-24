import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading image
img = cv2.imread('red_only_output.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Applying Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (7, 7), 0)
_, threshold = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Converting image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Applying Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Initialize variables for storing the largest triangle
max_triangle_area = 0
max_triangle = None

for contour in contours:
    # Approximating the shape
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Finding the center point of the shape
    M = cv2.moments(approx)
    if M['m00'] != 0:
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])

    # Detecting triangles
    if len(approx) == 3:
        # Calculate area of the triangle
        area = cv2.contourArea(approx)

        # Update the largest triangle if the current triangle has a greater area
        if area > max_triangle_area:
            max_triangle_area = area
            max_triangle = approx

# If a triangle is found, draw it on the image
if max_triangle is not None:
    cv2.drawContours(img, [max_triangle], 0, (0, 255, 0), -1)
    cv2.putText(img, 'Triangle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Display the image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Detected Largest Triangle')
plt.axis('off')
plt.show()
