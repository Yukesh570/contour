import cv2
import numpy as np

# Function to check if point (x, y) is inside the contour
def is_inside_contour(contour, point):
    return cv2.pointPolygonTest(contour, point, False) >= 0

# Function to draw the contour on the image
def draw_contour(image, contour):
    cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)

# Main function
def main():
    # Path to the video file
    video_path = r'C:\Users\Yukesh\Downloads\snookervideo\aja7.mp4'

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define range of green color in HSV
        lower_green = np.array([45, 100, 100])
        upper_green = np.array([75, 255, 255])

        # Threshold the HSV image to get only green colors
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if any green object touches the contour
        for cnt in contours:
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(cnt)

            # Calculate triangle vertices relative to the bounding box
            triangle_points = np.array([
                [x + w // 2, y],        # Top vertex
                [x + w, y + h],         # Right vertex
                [x, y + h]              # Left vertex
            ])

            # Draw the triangle
            cv2.polylines(frame, [triangle_points], isClosed=True, color=(0, 255, 0), thickness=2)

            # Draw the contour
            draw_contour(frame, cnt)

            # Calculate centroid of the contour
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # Check if centroid is inside the triangle
                if is_inside_contour(triangle_points, (cx, cy)):
                    cv2.putText(frame, "PLAY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Frame', frame)

        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
