import cv2
import numpy as np

cap = cv2.VideoCapture(r'C:\Users\Yukesh\Downloads\snookervideo\viber8.mp4')
# Define the lower and upper bounds for red color in HSV space
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])




def detectRedObjects(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    return red_mask

def detectObjects(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 220])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    return mask

def contour_touching(contour1,contour2):
    if contour2 is None:
        return False

    for point1 in contour1:
        for point2 in contour2:
                point1 = np.array(point1)
                point2 = np.array(point2)
                distance = np.linalg.norm(point1-point2)
                print('point1:', point1, 'point2:', point2, 'distance:', distance)
                if distance < 0.1:
                    return True
    return False
# def checkIntersection(contours, mask):
#     # Iterate through each contour
#     for cnt in contours:
#         # Iterate through each point in the contour
#         for point in cnt:
#             # Check if the point in the contour matches green color in the mask
#             if mask[point[0][1], point[0][0]] > 255:
#                 return True  # Intersection detected, return True
#     return False  # No intersection found, return False
def nothing(x):
    pass
# cv2.namedWindow('Trackbars')
# cv2.createTrackbar('Lower','Trackbars',20,255,nothing)
# cv2.createTrackbar('Upper','Trackbars',145,255,nothing)
#
skip_frames = 5
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Skip frames
    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    #retreive values
    frame=cv2.resize(frame,(500,500))
    # l = cv2.getTrackbarPos('Lower','Trackbars')
    # u = cv2.getTrackbarPos('Upper', 'Trackbars')
    l=20
    u=145
    # print(l,u)
    # imgBlur = cv2.GaussianBlur(frame, (25, 25), 1)

    # dst = cv2.fastNlMeansDenoisingColored(frame,None,15,15,3,10)
    red_mask=detectRedObjects(frame)
    mask=detectObjects(frame)
    median_blur= cv2.medianBlur(red_mask,11)
    median_blur_white= cv2.medianBlur(mask,11)


    # imgGray = cv2.cvtColor(median_blur, cv2.COLOR_BGR2GRAY)
    canny=cv2.Canny(median_blur,l,u)
    canny2=cv2.Canny(median_blur_white,l,u)

    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, hierarchy = cv2.findContours(canny2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_copy=frame.copy()
    cv2.drawContours(frame,contours,-1,(255,0,0),5)
    cv2.drawContours(frame_copy,contours2,-1,(255,0,0),10)

    # Check if the frame is not empty
    if frame is not None:
        # cv2.imshow('binary video',dst)
        # cv2.imshow('median video',median_blur_white)
        # if checkIntersection(contours, median_blur_white):
        #     cv2.putText(frame, "PLAY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if contour_touching(frame, frame_copy):
            cv2.putText(frame, "PLAY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Input', frame)
        # cv2.imshow('blur', imgGray)
        # cv2.imshow('binary video',canny2)

        # cv2.imshow('Input', frame_copy)
        # cv2.imshow('red', median_blur)

        # cv2.imshow('white', median_blur_white)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
