import cv2
import numpy as np

# Initial values for threshold
threshold1 = 1
threshold2 = 50

# Define the lower and upper bounds for red color in HSV space
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

def preprocess_frame(frame):
    frameBlur = cv2.GaussianBlur(frame, (7, 7), 1)
    frameGray = cv2.cvtColor(frameBlur, cv2.COLOR_BGR2GRAY)
    return frameGray

# Function to detect red objects
def detectRedObjects(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    return red_mask

# Function to detect green objects
def detectGreenObjects(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return mask

def checkIntersection(contours, mask):
    # Iterate through each contour
    for cnt in contours:
        # Iterate through each point in the contour
        for point in cnt:
            # Check if the point in the contour matches green color in the mask
            if mask[point[0][1], point[0][0]] > 100:
                return True  # Intersection detected, return True
    return False  # No intersection found, return False

# Function to update threshold1 value
def update_threshold1(value):
    global threshold1
    threshold1 = value

# Function to update threshold2 value
def update_threshold2(value):
    global threshold2
    threshold2 = value

# Read the video
video_path = r'C:\Users\Yukesh\Downloads\snookervideo\viber9.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit()

# Create a window and trackbars for threshold values
cv2.namedWindow('Frame')
cv2.createTrackbar('Threshold1', 'Frame', threshold1, 255, update_threshold1)
cv2.createTrackbar('Threshold2', 'Frame', threshold2, 255, update_threshold2)

# Process frames until the user presses 'q' to quit
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    frameGray = preprocess_frame(frame)

    # Detect red objects
    red_mask = detectRedObjects(frame)

    # Get threshold values from trackbars
    threshold1 = cv2.getTrackbarPos('Threshold1', 'Frame')
    threshold2 = cv2.getTrackbarPos('Threshold2', 'Frame')

    # Apply Canny edge detection
    frameCanny = cv2.Canny(red_mask, threshold1, threshold2)

    kernel = np.ones((5, 5))
    frameDil = cv2.dilate(frameCanny, kernel, iterations=1)

    # Detect green objects
    green_mask = detectGreenObjects(frame)

    # Check if any green object touches any contour
    contours, _ = cv2.findContours(frameDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if checkIntersection(contours, green_mask):
        cv2.putText(frame, "PLAY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw contours on the frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
