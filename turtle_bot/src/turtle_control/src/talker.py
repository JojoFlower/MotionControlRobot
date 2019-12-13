#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose

PI = 3.1415926535897


def poseCallback(pose_message):
    x = pose_message.x
    y = pose_message.y
    theta = pose_message.theta
    print("x = {}".format(x))
    print("y = {}".format(y))
    print("theta = {}".format(theta))


if __name__ == "__main__":
    try:
        rospy.init_node("turtle_control", anonymous=True)

        # declare velocity publisher
        cmd_vel_topic = "/turtle1/cmd_vel"
        velocity_publisher = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)

        position_topic = "/turtle1/pose"
        pose_subscriber = rospy.Subscriber(position_topic, Pose, poseCallback)

        loop_rate = rospy.Rate(0.5)

        while not rospy.is_shutdown():
            velocity_message = Twist()

            velocity_message.linear.x = 1
            velocity_publisher.publish(velocity_message)
            loop_rate.sleep()

            velocity_message.angular.z = PI/2
            velocity_publisher.publish(velocity_message)
            loop_rate.sleep()

            velocity_message.linear.x = -1
            velocity_publisher.publish(velocity_message)
            loop_rate.sleep()

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("node terminated.")
