#!/usr/bin/env python3
import rospy
import cv2 as cv
import numpy as np
import classifier as cl
from keras.models import model_from_json
import image_preprocessing as ip
from geometry_msgs.msg import Twist

PI = 3.1415926535897


if __name__ == "__main__":
    try:
        rospy.init_node("turtle_control", anonymous=True)
        loop_rate = rospy.Rate(0.5)

        # declare velocity publisher
        cmd_vel_topic = "/turtle1/cmd_vel"
        velocity_publisher = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)

        # Image acquisition
        cap = cv.VideoCapture(0)

        # load json and create model
        json_file = open("./networks/Conv8Conv4Drop05Dense32.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights("./networks/Conv8Conv4Drop05Dense32E150B32.h5")
        print("Loaded model from disk")

        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        while not rospy.is_shutdown():
            # Capture frame-by-frame
            ret, frame = cap.read()

            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Resize the image by x%, can be usefull for a faster runtime
            scale_percent = 60  # percent of original size
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv.resize(frame, dim, interpolation=cv.INTER_AREA)

            velocity_message = Twist()

            err, coeff = ip.live_preprocessing(frame)
            if not err:
                prediction = loaded_model.predict(np.array([coeff]), verbose=1)
                output = cl.max_prediction(prediction)
                print(output)

                if output == 'open_hand':
                    velocity_message.linear.x = 1
                    velocity_publisher.publish(velocity_message)
                if output == 'closed_hand':
                    velocity_message.linear.x = 0
                    velocity_publisher.publish(velocity_message)
                if output == 'tight_hand':
                    velocity_message.angular.z = PI/2
                    velocity_publisher.publish(velocity_message)
                if output == 'side_hand':
                    velocity_message.angular.z = -PI/2
                    velocity_publisher.publish(velocity_message)

            else:
                print("Error while calculating fourrier descriptors")

            # Display the resulting frame
            cv.imshow("frame", frame)

            if cv.waitKey(1) == ord("q"):
                break

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("node terminated.")
