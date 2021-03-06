#!/usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError


def main():
    rospy.init_node("video_publisher", anonymous=True)
    img_pub = rospy.Publisher("/image_raw", Image, queue_size=10)
    bridge = CvBridge()
    video = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('/home/ubuntu/disk/out.avi', fourcc, 30.0, (1980, 1080))
    fps = video.get(cv2.CAP_PROP_FPS)
    rate = rospy.Rate(fps)
    while not rospy.is_shutdown() and video.grab():
        tmp, img = video.retrieve()

        if not tmp:
            print("Could not grab frame.")
            break

        try:
            img = cv2.flip(img, -1)
            img = cv2.blur(img, (5, 5))
            out.write(img)
            # Publish image.
            img_msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")
            img_msg.header.stamp = rospy.Time.now()
            img_pub.publish(img_msg)

        except CvBridgeError as err:
            print(err)

        rate.sleep()
    return


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
