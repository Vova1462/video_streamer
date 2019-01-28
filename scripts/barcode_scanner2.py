from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2
from imutils.video import VideoStream
import time
import imutils


def transform_image(image, K, D, DIM):
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


def detect(image):
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    grad_x = cv2.Sobel(image, ddepth, dx=1, dy=0, ksize=-1)
    grad_y = cv2.Sobel(image, ddepth, dx=0, dy=1, ksize=-1)

    gradient = cv2.subtract(grad_x, grad_y)
    gradient = cv2.convertScaleAbs(gradient)

    blured = cv2.blur(gradient, (9, 9))
    (_, treshold) = cv2.threshold(blured, 255, 255, cv2.THRESH_BINARY)

    return box


def decode(im):
    # Find barcodes and QR codes
    decodedObjects = pyzbar.decode(im)

    # Print results
    for obj in decodedObjects:
        print('Type : ', obj.type)
        print('Data : ', obj.data, '\n')

    return decodedObjects


# Display barcode and QR code location  
def display(im, decodedObjects):
    # Loop over all decoded objects
    for decodedObject in decodedObjects:
        points = decodedObject.polygon

        # If the points do not form a quad, find convex hull
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else:
            hull = points

        # Number of points in the convex hull
        n = len(hull)

        # Draw the convext hull
        for j in range(0, n):
            cv2.line(im, hull[j], hull[(j + 1) % n], (255, 0, 0), 3)

    # Display results
    cv2.imshow("Results", im)
    # cv2.waitKey(0)


# Main 
if __name__ == '__main__':
    # Read image
    # vs = VideoStream(src=2).start()
    vs = cv2.VideoCapture(2)
    vs.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    vs.set(cv2.CAP_PROP_FOCUS, 0.2)
    vs.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    time.sleep(2.0)
    #  csv = open('/home/vladimir/Documents/barcodes.csv', 'w')
    found = set()
    # DIM, K, D = np.load('calibration.npy', encoding='bytes')

    while True:
        ret, image = vs.read()
        im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        box = detect(image)
        #im = transform_image(im, K, D, DIM)
        # im = imutils.resize(im, width=400)
        decodedObjects = decode(image)
        display(image, decodedObjects)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
    # vs.stop()
