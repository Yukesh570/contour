import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image_path = r'C:\Users\Yukesh\Downloads\snookervideo\custom2.JPG'
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read the image.")
    exit()

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range for red color in HSV, including darker shades
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 50, 50])
upper_red2 = np.array([180, 255, 255])

# Define the range for white color in HSV
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 25, 255])

# Create masks to detect red regions
mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask_red1, mask_red2)

# Create a mask to detect white regions
white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

# Combine the red and white masks
combined_mask = cv2.bitwise_or(red_mask, white_mask)

# Optionally, you can apply some additional filtering to reduce false positives
# Perform morphology operations like opening and closing
kernel = np.ones((5, 5), np.uint8)
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

# Use the combined mask to keep only the red and white regions in the original image
red_and_white_regions = cv2.bitwise_and(image, image, mask=combined_mask)

# Save the final image
output_path = r'C:\Users\Yukesh\Downloads\snookervideo\red_and_white_output.jpg'
cv2.imwrite(output_path, red_and_white_regions)
print(f"Processed image saved to {output_path}")

# Display the images using matplotlib
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# plt.subplot(1, 3, 2)
# plt.title('Red Mask')
# plt.imshow(red_mask, cmap='gray')
# plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Red and White Image')
plt.imshow(cv2.cvtColor(red_and_white_regions, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
