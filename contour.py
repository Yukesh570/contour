import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to the local image
image_path = r'C:\Users\Yukesh\Downloads\snookervideo\custom1red.jpg'
image = cv2.imread(image_path)  # Replace with the path to your local image

# Initial values for threshold
threshold1 = 30
threshold2 = 30

# Define the lower and upper bounds for white color in HSV space
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 25, 255])

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)

    if rowsAvailable:
        height = imgArray[0][0].shape[0]
        width = imgArray[0][0].shape[1]
    else:
        height = imgArray[0].shape[0]
        width = imgArray[0].shape[1]

    for x in range(rows):
        for y in range(cols):
            img = imgArray[x][y]
            if img.shape[:2] != (height, width):
                imgArray[x][y] = cv2.resize(img, (width, height))
            if len(img.shape) == 2:  # If the image is grayscale
                imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)

    if rowsAvailable:
        hor = [np.hstack(row) for row in imgArray]
        ver = np.vstack(hor)
    else:
        ver = np.hstack(imgArray)

    return ver

def preprocess_image(img):
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    return imgGray

def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50000 > area > 500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 6)  # Increased thickness to 2
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)


            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Increased thickness to 2


            # Adjusted position of text to top-right corner of rectangle
            cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w - 60, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.3,
                        (0, 255, 0), 1)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + w - 60, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.3,
                        (0, 255, 0), 1)

def outlinecontour(img, imgContour2):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(imgContour2)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 6)
    return contours, contour_img

# Function to detect white objects
def detectWhiteObjects(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    return mask

# Function to check if a white object touches any contour
def checkIntersection(contours, mask):
    for cnt in contours:
        for point in cnt:
            if mask[point[0][1], point[0][0]] == 255:
                return True
    return False

def update(val):
    global threshold1, threshold2  # Declare variables as global to update their values
    threshold1 = int(thresh1_slider.val)
    threshold2 = int(thresh2_slider.val)

    imgGray = preprocess_image(image)

    # Apply Canny edge detection
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)

    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

    imgContour = image.copy()
    imgContour2 = image.copy()
    getContours(imgDil, imgContour)
    contours, outline = outlinecontour(imgDil, imgContour2)

    # Detect white objects in the image
    white_objects_mask = detectWhiteObjects(image)

    # Check if any white object touches any contour
    if checkIntersection(contours, white_objects_mask):
        print("Game start")

    imgStack = stackImages(2, [[image, imgGray, imgCanny, imgDil, imgContour, outline]])

    # Display the image using Matplotlib
    ax.clear()
    ax.imshow(cv2.cvtColor(imgStack, cv2.COLOR_BGR2RGB))
    plt.draw()

# Read the local image
if image is None:
    print(f"Error: Could not read image from {image_path}")
    exit()

# Increase the size of the Matplotlib window
plt.rcParams['figure.figsize'] = [12, 10]

# Create a figure and a set of subplots
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

# Create sliders for threshold adjustments
axcolor = 'lightgoldenrodyellow'
ax_thresh1 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_thresh2 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

thresh1_slider = plt.Slider(ax_thresh1, 'Threshold1', 0, 255, valinit=threshold1, valstep=1)
thresh2_slider = plt.Slider(ax_thresh2, 'Threshold2', 0, 255, valinit=threshold2, valstep=1)

thresh1_slider.on_changed(update)
thresh2_slider.on_changed(update)

# Initial display
update(0)

# Show the plot
plt.show()
