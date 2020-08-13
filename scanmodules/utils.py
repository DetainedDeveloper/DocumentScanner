import cv2
import numpy as np

t_bars = 'Trackbars'
thre_1 = 'Threshold 1'
thre_2 = 'Threshold 2'


class ScanUtils:

    @staticmethod
    def begin():
        print('Created by https://github.com/DetainedDeveloper?tab=repositories')

    # Display all images in one window
    @classmethod
    def displayAllImages(cls, image_array, scale, show_labels):

        # Get total number of rows and columns
        rows = len(image_array)
        columns = len(image_array[0])

        rows_available = isinstance(image_array[0], list)

        # Get width and height of original image
        width = image_array[0][0].shape[1]
        height = image_array[0][0].shape[0]

        if rows_available:

            for row in range(rows):
                for col in range(columns):

                    # Resize every image in image array
                    image_array[row][col] = cv2.resize(image_array[row][col], (0, 0), None, scale, scale)

                    if len(image_array[row][col].shape) == 2:
                        image_array[row][col] = cv2.cvtColor(image_array[row][col], cv2.COLOR_GRAY2BGR)

            # Create a blank image which will be displayed when there is nothing to show
            image_blank = np.zeros((height, width, 3), np.uint8)

            # Horizontal and Vertical concatenation

            image_horizontal = [image_blank] * rows
            image_horizontal_con = [image_blank] * rows

            for row in range(rows):
                image_horizontal[row] = np.hstack(image_array[row])
                image_horizontal_con[row] = np.concatenate(image_array[row])

            image_vertical = np.vstack(image_horizontal)
            # image_vertical_con = np.concatenate(image_horizontal)

        else:

            for row in range(rows):
                image_array[row] = cv2.resize(image_array[row], (0, 0), None, scale, scale)

                if len(image_array[row].shape) == 2:
                    image_array[row] = cv2.cvtColor(image_array[row], cv2.COLOR_GRAY2BGR)

            image_horizontal = np.hstack(image_array)
            # image_horizontal_con = np.concatenate(image_array)

            image_vertical = image_horizontal

        if show_labels:
            cls.show_labels(rows, columns, image_vertical)

        return image_vertical

    @staticmethod
    def show_labels(rows, columns, image_vertical):

        display_labels = [['Original', 'Gray', 'Threshold', 'All Contours'],
                          ['Biggest Contour', 'Wrap Perspective', 'Wrap Gray', 'Adaptive Threshold']]

        final_width = int(image_vertical.shape[1] / columns)
        final_height = int(image_vertical.shape[0] / rows)
        
        for row in range(rows):
            for col in range(columns):

                # Draw a white rectangle as background of label
                cv2.rectangle(image_vertical, (final_width * col, final_height * row),
                              (final_width * col + len(display_labels[row][col]) * 10 + 20, final_height * row + 20),
                              (255, 0, 127), cv2.FILLED)

                # Draw label text
                cv2.putText(image_vertical, display_labels[row][col],
                            (final_width * col, final_height * row + 15),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)

    # Reorder all points
    @staticmethod
    def reorder(points):

        points = points.reshape((4, 2))
        new_points = np.zeros((4, 1, 2), dtype=np.int32)

        add = points.sum(1)

        new_points[0] = points[np.argmin(add)]
        new_points[3] = points[np.argmax(add)]

        difference = np.diff(points, axis=1)

        new_points[1] = points[np.argmin(difference)]
        new_points[2] = points[np.argmax(difference)]

        return new_points

    # Find the biggest contour (Hopefully, the page/document)
    @staticmethod
    def findBiggestContour(contours):

        biggest_contour = np.array([])
        max_area = 0

        for cont in contours:

            area = cv2.contourArea(cont)

            if area > 5000:
                per = cv2.arcLength(cont, True)
                approx = cv2.approxPolyDP(cont, 0.02 * per, True)

                if area > max_area and len(approx) == 4:
                    biggest_contour = approx
                    max_area = area

        return biggest_contour, max_area

    # Draw a rectangle around the biggest contour
    @staticmethod
    def drawRectangle(image, biggest_contour, thickness):

        cv2.line(image, (biggest_contour[0][0][0], biggest_contour[0][0][1]),
                        (biggest_contour[1][0][0], biggest_contour[1][0][1]),
                 (0, 255, 255), thickness)

        cv2.line(image, (biggest_contour[0][0][0], biggest_contour[0][0][1]),
                        (biggest_contour[2][0][0], biggest_contour[2][0][1]),
                 (0, 255, 255), thickness)

        cv2.line(image, (biggest_contour[3][0][0], biggest_contour[3][0][1]),
                        (biggest_contour[2][0][0], biggest_contour[2][0][1]),
                 (0, 255, 255), thickness)

        cv2.line(image, (biggest_contour[3][0][0], biggest_contour[3][0][1]),
                        (biggest_contour[1][0][0], biggest_contour[1][0][1]),
                 (0, 255, 255), thickness)

        return image

    # Initialize threshold trackbar window
    @classmethod
    def initializeTrackbars(cls):

        cv2.namedWindow(t_bars)
        cv2.resizeWindow(t_bars, 360, 240)
        cv2.createTrackbar(thre_1, t_bars, 150, 255, cls.nothing)
        cv2.createTrackbar(thre_2, t_bars, 150, 255, cls.nothing)

    # Initialize default value for threshold trackbars
    @staticmethod
    def trackbarValues():

        threshold_1 = cv2.getTrackbarPos(thre_1, t_bars)
        threshold_2 = cv2.getTrackbarPos(thre_2, t_bars)
        return threshold_1, threshold_2

    # Just, nothing!
    @staticmethod
    def nothing():
        pass
