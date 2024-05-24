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
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Create masks to detect red regions
mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)

# Optionally, you can apply some additional filtering to reduce false positives
# For example, you can perform morphology operations like opening and closing
# This helps in removing small noise and filling gaps in the detected regions
kernel = np.ones((5, 5), np.uint8)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

# Create an inverted mask for the non-red regions
non_red_mask = cv2.bitwise_not(red_mask)

# Make a black image of the same size as the original
black_image = np.zeros_like(image)

# Use the mask to keep only the red regions in the original image
red_regions = cv2.bitwise_and(image, image, mask=red_mask)

# Combine the red regions with the black image
final_image = cv2.bitwise_or(red_regions, black_image, mask=red_mask)

# Display the images using matplotlib
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Red Only Image')
plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()

# Save the final image
output_path = r'C:\Users\Yukesh\Downloads\snookervideo\red_only_output.jpg'
cv2.imwrite(output_path, final_image)
print(f"Processed image saved to {output_path}")
