import cv2
import numpy as np
from scanmodules import utils

scan_utils = utils.ScanUtils()


class Scanner:

    def __init__(self):
        scan_utils.begin()

    # Start OpenCV camera feed
    @classmethod
    def cam(cls, cam_number):

        capture = cv2.VideoCapture(cam_number)
        # capture.set(cv2.CAP_PROP_FPS, 24)
        # capture.set(10, 160)

        scan_utils.initializeTrackbars()

        while capture.isOpened():

            ret, frame = capture.read()

            scanned = cls.scan(frame)
            result = scan_utils.displayAllImages(scanned, 0.75, True)

            cv2.imshow('Scanner', result)

            # Detect a keypress
            key = cv2.waitKey(1)

            # Esc == 27
            if key == 27:
                cv2.destroyAllWindows()
                break

    # Scan image
    @classmethod
    def scan(cls, image):

        scan_utils.initializeTrackbars()

        width = 640
        height = 480

        # Resize image
        image = cv2.resize(image, (width, height))

        # Create a blank image
        image_blank = np.zeros((height, width, 3), np.uint8)

        # Covert image to Grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        image_blur = cv2.GaussianBlur(image_gray, (5, 5), 1)

        # Get threshold values from threshold trackbars
        threshold_values = scan_utils.trackbarValues()

        # Perform Canny Edge Detection
        image_canny = cv2.Canny(image_blur, threshold_values[0], threshold_values[1])

        kernel = np.ones((5, 5))

        # Apply Dilation
        image_dilate = cv2.dilate(image_canny, kernel, iterations=2)

        # Apply Erosion
        image_canny = cv2.erode(image_dilate, kernel, iterations=1)

        # Create image to draw all detected contours
        image_all_contours = image.copy()

        # Create image to draw the biggest contour
        image_biggest_contour = image.copy()

        # Find all contours
        all_contours, hierarchy = cv2.findContours(image_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw all detected contours
        cv2.drawContours(image_all_contours, all_contours, -1, (0, 255, 0), 10)

        biggest_contour, max_area = scan_utils.findBiggestContour(all_contours)

        if biggest_contour.size != 0:
            # Get the biggest contour
            return cls.biggestContour(image, image_gray, image_canny, image_all_contours,
                                      image_biggest_contour, biggest_contour, width, height)

        else:
            return ([image, image_gray, image_canny, image_all_contours],
                    [image_blank, image_blank, image_blank, image_blank])

    # Find biggest contour
    @classmethod
    def biggestContour(cls, image, image_gray, image_canny, image_all_contours,
                       image_biggest_contour, biggest_contour, width, height):

        biggest_contour = scan_utils.reorder(biggest_contour)

        # Draw the biggest contour
        cv2.drawContours(image_biggest_contour, biggest_contour, -1, (255, 255, 255), 20)

        # Draw a rectangle around biggest contour
        image_biggest_contour = scan_utils.drawRectangle(image_biggest_contour, biggest_contour, 3)

        return cls.perspective(image, image_gray, image_canny, image_all_contours,
                               image_biggest_contour, biggest_contour, width, height)

    # Create perspective images
    @staticmethod
    def perspective(image, image_gray, image_canny, image_all_contours,
                    image_biggest_contour, biggest_contour, width, height):

        # Prepare points to create a warp perspective image
        p1 = np.float32(biggest_contour)
        p2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        perspective = cv2.getPerspectiveTransform(p1, p2)

        # Create a coloured wrap perspective image
        image_wrap_coloured = cv2.warpPerspective(image, perspective, (width, height))

        # Create a Grayscale warp perspective image
        image_warp_gray = cv2.cvtColor(image_wrap_coloured, cv2.COLOR_BGR2GRAY)

        # Apply adaptive threshold
        image_adaptive_threshold = cv2.adaptiveThreshold(image_warp_gray, 255, 1, 1, 7, 2)
        image_adaptive_threshold = cv2.bitwise_not(image_adaptive_threshold)
        image_adaptive_threshold = cv2.medianBlur(image_adaptive_threshold, 3)

        return ([image, image_gray, image_canny, image_all_contours],
                [image_biggest_contour, image_wrap_coloured, image_warp_gray, image_adaptive_threshold])
