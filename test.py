import cv2
import numpy as np

# Function to detect white objects
def detectWhiteObjects(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    return mask

# Function to check if a white object touches any contour
def checkIntersection(contours, mask):
    for cnt in contours:
        for point in cnt:
            if mask[point[0][1], point[0][0]] == 255:
                return True
    return False

# Read the video
video_path = r'C:\Users\Yukesh\Downloads\snookervideo\aja2.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit()

# Define the lower and upper bounds for red color in HSV space
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 50, 50])
upper_red2 = np.array([180, 255, 255])

# Skip frames
skip_frames = 5
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames
    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks to detect red regions
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    # Apply Gaussian blur to further reduce noise
    red_mask_blurred = cv2.GaussianBlur(red_mask, (9, 9), 2)

    # Find contours of the red regions
    contours, _ = cv2.findContours(red_mask_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw smooth contours around the red regions
    frame_contour = frame.copy()
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(frame_contour, [approx], -1, (0, 0, 255), 2)

    # Detect white objects in the frame
    white_objects_mask = detectWhiteObjects(frame)

    # Check if any white object touches any contour
    if checkIntersection(contours, white_objects_mask):
        print("Game started")

    # Display the frame with red contours using cv2.imshow
    cv2.imshow('Frame', frame_contour)

    # Press 'q' to exit the loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
