import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to the local image
image_path = r'C:\Users\Yukesh\Downloads\snookervideo\balls.jpg'
image = cv2.imread(image_path)  # Replace with the path to your local image

# Initial values for threshold
threshold1 = 30
threshold2 = 30

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
    imgBlur = cv2.GaussianBlur(img, (5, 5), 5)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    return imgGray

def is_curve(points, angle_threshold=60):
    angles = []
    for i in range(len(points)):
        p1 = points[i - 1]
        p2 = points[i]
        p3 = points[(i + 1) % len(points)]

        v1 = p1 - p2
        v2 = p3 - p2

        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle = np.degrees(angle)
        angles.append(angle)

    # Detecting curves: if the angle is less than the threshold, it's a straight edge; otherwise, it's a curve
    is_curve = np.array(angles) > angle_threshold
    return is_curve

def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    total_curves = 0  # Initialize the count of curves

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 20000 > area > 2000:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 2)  # Increased thickness to 2
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            approx_points = np.squeeze(approx, axis=1)
            curve_mask = is_curve(approx_points)

            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Increased thickness to 2

            # Count the number of curves
            total_curves += np.sum(curve_mask)

            # Draw circles on curve points
            for i, is_curve_point in enumerate(curve_mask):
                color = (0, 0, 255) if is_curve_point else (255, 0, 0)
                cv2.circle(imgContour, tuple(approx_points[i]), 5, color, -1)

            # Adjusted position of text to top-right corner of rectangle
            cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w - 60, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.3,
                        (0, 255, 0), 1)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + w - 60, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.3,
                        (0, 255, 0), 1)

            # Display the number of curves on the top-right corner
            cv2.putText(imgContour, "Curves: " + str(total_curves), (x + w - 60, y + 70),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6,
                        (0, 255, 255), 1)

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
    getContours(imgDil, imgContour)
    imgStack = stackImages(3, [[image, imgGray, imgCanny, imgDil, imgContour]])

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
