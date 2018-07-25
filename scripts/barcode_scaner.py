import cv2
from pyzbar import pyzbar
import imutils
from imutils.video import VideoStream
import time
from datetime import datetime
import numpy as np


def transform_image(image, K, D, DIM):
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


if __name__=='__main__':
    # cap = cv2.VideoCapture(1)
    vs = VideoStream(src=1).start()
    time.sleep(2.0)
    csv = open('/home/vladimir/Documents/barcodes.csv', 'w')
    found = set()
    DIM, K, D = np.load('/home/vladimir/PycharmProjects/barcode_saner/calibration.npy', encoding='bytes')

    while True:
        frame = vs.read()
        #frame = imutils.resize(frame, width=400)
        frame = transform_image(frame, K, D, DIM)

        barcodes = pyzbar.decode(frame)

        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            barcode_data = barcode.data.decode('utf-8')
            barcode_type = barcode.type

            text = "{}({})".format(barcode_data, barcode_type)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if barcode_data not in found:
                csv.write("{},{}\n".format(datetime.now(), barcode_data))
                csv.flush()
                found.add(barcode_data)

        cv2.imshow('Image', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    csv.close()
    cv2.destroyAllWindows()
    vs.stop()
