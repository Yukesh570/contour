import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading image
img = cv2.imread('images3.jpg')

# Converting image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Applying Gaussian blur to redquce noise
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# Setting threshold of the gray image
_, threshold = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)

# Finding contours
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize variable to count circles
circle_count = 0

# List to store detected circle centers
circle_centers = []
triangle_count = 0

# List for storing names of shapes
for contour in contours:
    # Approximating the shape
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Finding the center point of the shape
    M = cv2.moments(approx)
    if M['m00'] != 0:
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])

    # Putting shape name at the center of each shape
    if len(approx) == 3:
        cv2.drawContours(img, [approx], 0, (0, 255, 0), -1)
        cv2.putText(img, 'Triangle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        # Using Hough Circle Transform to detect circles
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=40,
                                   param1=30, param2=20, minRadius=15, maxRadius=44)

        if circles is not None:
            # Converting circles to integer coordinates
            circles = np.uint16(np.around(circles))

            # Drawing circles and counting unique circles
            for i in circles[0, :]:
                center = (i[0], i[1])
                if center not in circle_centers:
                    circle_centers.append(center)
                    circle_count += 1
                    cv2.circle(img, center, i[2], (0, 0, 255), 2)
                    cv2.putText(img, 'Circle', center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Check if there are at least 3 circles detected
        if len(circles) >= 3:
            # Convert circle centers to numpy array
            centers = circles[:, :2]

            # Calculate pairwise distances between circle centers
            dists = np.sqrt(((centers[:, None] - centers) ** 2).sum(axis=2))

            # Define a minimum distance to consider valid triangles
            min_triangle_side = 50

            # Check if distances satisfy the geometric conditions of a triangle
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    for k in range(j + 1, len(centers)):
                        side_lengths = sorted([dists[i, j], dists[i, k], dists[j, k]])
                        # Check if the three distances form a valid triangle
                        if side_lengths[0] + side_lengths[1] > side_lengths[2] and side_lengths[0] > min_triangle_side:
                            # Increment triangle count
                            triangle_count += 1
                            # Draw lines between circle centers to form a triangle
                            cv2.line(img, (centers[i][0], centers[i][1]), (centers[j][0], centers[j][1]), (0, 0, 255),
                                     2)
                            cv2.line(img, (centers[j][0], centers[j][1]), (centers[k][0], centers[k][1]), (0, 0, 255),
                                     2)
                            cv2.line(img, (centers[k][0], centers[k][1]), (centers[i][0], centers[i][1]), (0, 0, 255),
                                     2)

    # Print the number of triangles detected
    print("Number of triangles detected:", triangle_count)
# Print the number of circles detected
print("Number of circles detected:", circle_count)

# Display the image using Matplotlib
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Detected Shapes')
plt.axis('off')
plt.show()
#
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Read the image
# img = cv2.imread('triangle.jpg')
#
# # Check if the image is loaded successfully
# if img is None:
#     print("Error: Unable to load the image.")
# else:
#     # Convert image to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # Apply Gaussian blur to reduce noise
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#
#     # Thresholding
#     _, threshold = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)
#
#     # Find contours
#     contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Initialize counter for triangles detected
#     triangle_count = 0
#
#     # Iterate through contours
#     for contour in contours:
#         # Approximate the shape
#         epsilon = 0.02 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)
#
#         # Filter contours based on the number of vertices (triangles have 3 vertices)
#         if len(approx) == 3:
#             # Draw contours on the original image
#             cv2.drawContours(img, [contour], 1, (0, 255, 0), 2)
#             # Increment triangle count
#             triangle_count += 1
#
#     # Put the number of triangles detected on the image
#     cv2.putText(img, f'Triangles Detected: {triangle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#
#     # Display the image with detected triangles
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.show()
#     cv2.destroyAllWindows()
#
