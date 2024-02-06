import cv2 as cv
import numpy as np
from time import sleep

# Minimum width and height for rectangle detection
min_width = 80
min_height = 80

# Offset allowed between pixels
offset = 6

# Y-position of the counting line
line_position = 550

# Frames per second for video
fps = 60

detections = []
cars_left = 0

cars_right = 0 

def get_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

cap = cv.VideoCapture('video.mp4')
background_subtractor = cv.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame1 = cap.read()
    sleep(1/fps)

    # Convert the frame to grayscale and apply Gaussian blur
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    blur1 = cv.GaussianBlur(gray1, (3, 3), 5)

    # Apply background subtraction
    img_sub1 = background_subtractor.apply(blur1)

    # Perform dilation
    dilated1 = cv.dilate(img_sub1, np.ones((5, 5)))

    # Define the morphological kernel
    kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

    # Apply morphological transformations
    dilated1 = cv.morphologyEx(dilated1, cv.MORPH_CLOSE, kernel1)
    dilated1 = cv.morphologyEx(dilated1, cv.MORPH_CLOSE, kernel1)

    # Split the dilated images for counting
    left = dilated1[0:800,0:600]
    right = dilated1[0:800, 600:1200]

    # Find contours in the dilated left image
    contours1, hierarchy1 = cv.findContours(left, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Find contours in the dilated right image
    contours2, hierarchy2 = cv.findContours(right, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Draw the counting line
    #cv.line(frame1, (25, line_position), (1200, line_position), (255, 255, 0), 2)

    # Iterate through each contour for left side
    for (i, contour1) in enumerate(contours1):
        (x, y, w, h) = cv.boundingRect(contour1)
        
        # Check if the contour meets the minimum width and height criteria
        validate_contour = (w >= min_width) and (h >= min_height)
        if not validate_contour:
            continue

        # Draw a rectangle around the detected object
        cv.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get the center of the rectangle
        center = get_center(x, y, w, h)

        # Add the object into list
        detections.append(center)

        # Draw a circle at the center
        cv.circle(frame1, center, 4, (0, 0, 255), -1)

        # Check if the center of the object is near the counting line
        for (cx, cy) in detections:
            if line_position - offset < cy < line_position + offset:
                cars_left += 1
                #cv.line(frame1, (25, line_position), (1200, line_position), (0, 255, 255), 3)
                detections.remove((cx, cy))
                print("Incoming:  " + str(cars_left))
    
    # Iterate through each contour for right side
    for (i, contour2) in enumerate(contours2):
        (x, y, w, h) = cv.boundingRect(contour2)
        
        # Check if the contour meets the minimum width and height criteria
        validate_contour = (w >= min_width) and (h >= min_height)
        if not validate_contour:
            continue

        # Draw a rectangle around the detected object
        cv.rectangle(frame1, (x+600, y), (x + w+600, y + h), (0, 255, 0), 2)

        # Get the center of the rectangle
        center = get_center(x+600, y, w, h)

        # Add the object into list
        detections.append(center)

        # Draw a circle at the center
        cv.circle(frame1, center, 4, (0, 0, 255), -1)

        # Check if the center of the object is near the counting line
        for (cx, cy) in detections:
            if line_position - offset < cy < line_position + offset:
                cars_right += 1
                #cv.line(frame1, (25, line_position), (1200, line_position), (0, 255, 255), 3)
                detections.remove((cx, cy))
                print("Outgoing:  " + str(cars_right))

    # Display the vehicle count on the frame
    cv.putText(frame1, "Entering: " + str(cars_right), (700, 70), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    # Display the vehicle count on the frame
    cv.putText(frame1, "Exiting: " + str(cars_left), (25, 70), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    # Show the original and processed frames
    cv.imshow("Original Video", frame1)
    # cv.imshow("Detection", dilated)

    # Break the loop if the 'Esc' key is pressed
    if cv.waitKey(1) == 27:
        break

# Close all windows and release the video capture object
cv.destroyAllWindows()
cap.release()
