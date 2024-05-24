import cv2
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def detect_largest_triangle(image_path, min_triangle_area=1000):
    # Load the image
    image = cv2.imread("ball4.jpg")

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Function to update the detected circles based on the slider values
    def update_circles(val):
        min_dist = int(slider_min_dist.val)
        param1_val = int(slider_param1.val)
        param2_val = int(slider_param2.val)
        min_radius = int(slider_min_radius.val)
        max_radius = int(slider_max_radius.val)

        # Apply Hough Circle Transform to detect circles
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist, param1=param1_val, param2=param2_val,
                                   minRadius=min_radius, maxRadius=max_radius)

        # If circles are detected
        if circles is not None:
            circles = np.uint16(np.around(circles))
            centers = circles[0, :, :2]

            # Initialize variables to store the largest triangle found
            max_area = 0
            max_triangle = None

            # Generate combinations of circle centers to form triangles
            combinations = list(itertools.combinations(centers, 3))

            # Check if each combination forms a triangle
            for combination in combinations:
                # Calculate area of the triangle using cross product
                area = 0.5 * abs(np.cross(combination[1] - combination[0], combination[2] - combination[0]))

                # Update the largest triangle if the current triangle has a greater area
                if area > max_area and area > min_triangle_area:
                    max_area = area
                    max_triangle = combination

            # If a triangle with a valid area is found, draw it on the image
            if max_triangle is not None:
                cv2.line(image, tuple(max_triangle[0]), tuple(max_triangle[1]), (0, 255, 0), 2)
                cv2.line(image, tuple(max_triangle[1]), tuple(max_triangle[2]), (0, 255, 0), 2)
                cv2.line(image, tuple(max_triangle[2]), tuple(max_triangle[0]), (0, 255, 0), 2)

        # Convert BGR to RGB for displaying with matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the image
        ax.imshow(image_rgb)
        plt.axis('off')
        plt.show()

    # Create sliders for parameter adjustment
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.45)
    ax.imshow(gray, cmap='gray')

    # Slider for minDist
    ax_min_dist = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider_min_dist = Slider(ax_min_dist, 'minDist', 0, 200, valinit=40, valstep=1)

    # Slider for param1
    ax_param1 = plt.axes([0.25, 0.15, 0.65, 0.03])
    slider_param1 = Slider(ax_param1, 'param1', 0, 200, valinit=70, valstep=1)

    # Slider for param2
    ax_param2 = plt.axes([0.25, 0.2, 0.65, 0.03])
    slider_param2 = Slider(ax_param2, 'param2', 0, 200, valinit=60, valstep=1)

    # Slider for minRadius
    ax_min_radius = plt.axes([0.25, 0.25, 0.65, 0.03])
    slider_min_radius = Slider(ax_min_radius, 'minRadius', 0, 100, valinit=10, valstep=1)

    # Slider for maxRadius
    ax_max_radius = plt.axes([0.25, 0.3, 0.65, 0.03])
    slider_max_radius = Slider(ax_max_radius, 'maxRadius', 0, 200, valinit=60, valstep=1)

    # Attach update function to sliders
    slider_min_dist.on_changed(update_circles)
    slider_param1.on_changed(update_circles)
    slider_param2.on_changed(update_circles)
    slider_min_radius.on_changed(update_circles)
    slider_max_radius.on_changed(update_circles)

    plt.show()


# Example usage:
image_path = 'ball4.jpg'  # Replace 'ball4.jpg' with the path to your image
detect_largest_triangle(image_path)
