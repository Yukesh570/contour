import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image_path = r'C:\Users\Yukesh\Downloads\snookervideo\ball3.JPG'
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read the image.")
    exit()

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range for red color in HSV
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

# Create masks to detect red regions
mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)

# Create an inverted mask for the non-red regions
non_red_mask = cv2.bitwise_not(red_mask)

# Make a black image of the same size as the original
black_image = np.zeros_like(image)

# Use the mask to keep only the red regions in the original image
red_regions = cv2.bitwise_and(image, image, mask=red_mask)

# Combine the red regions with the black image
final_image = cv2.bitwise_or(red_regions, black_image, mask=red_mask)
image=final_image



# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Setting threshold value to get new image
# _, thresh_image = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)
#
# # Retrieving outer-edge coordinates in the new threshold image
# contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# # Iterating through each contour to retrieve coordinates of each shape
# for i, contour in enumerate(contours):
#     if i == 0:
#         continue
#
#     epsilon = 0.01 * cv2.arcLength(contour, True)
#     approx = cv2.approxPolyDP(contour, epsilon, True)
#
#     cv2.drawContours(image, contour, 0, (0, 0, 0), 4)
#
#     x, y, w, h = cv2.boundingRect(approx)
#     x_mid = int(x + (w / 3))
#     y_mid = int(y + (h / 1.5))
#
#     coords = (x_mid, y_mid)
#     colour = (0, 0, 0)
#     font = cv2.FONT_HERSHEY_DUPLEX
#
#     if len(approx) == 3:
#         cv2.putText(image, "Triangle", coords, font, 1, colour, 1)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Detect edges using Canny edge detector
edges = cv2.Canny(blurred_image, 50, 150)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through the contours
for contour in contours:
    # Approximate the contour to a polygon
    approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)

    # If the contour has 3 vertices, it's a triangle
    if len(approx) == 3:
        # Draw the triangle on the original image
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)


# Save the processed image
plt.figure(figsize=(10, 5))



plt.subplot(1, 2, 2)
plt.title('Red Only Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
cv2.imwrite("detected_shapes.jpg", image)