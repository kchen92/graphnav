from __future__ import print_function

import numpy as np
import os
import rospkg
import rospy
import yaml

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from semnav.config import get_config
from semnav.dataset.depth_transform import DepthTransform
from semnav.dataset.parsebag import BagParser
from sensor_msgs.msg import Image


class BehaviorExecutor(object):

    def __init__(self, node_name, mode):
        """Constructor.

        Args:
            mode: 'real' or 'sim', it will change which topics to subscribe.
        """
        self.cfg = get_config()
        self.bridge = CvBridge()
        self.mode = mode

        rospy.init_node(node_name)

        rospack = rospkg.RosPack()
        gibson_path = rospack.get_path('gibson-ros')
        ros_cfg_yaml_filepath = os.path.join(gibson_path, 'turtlebot_rgbd.yaml')
        with open(ros_cfg_yaml_filepath, 'r') as f:
            ros_cfg = yaml.load(f)

        self.depth_transform = DepthTransform(min_depth=self.cfg.min_depth,
                                              max_depth=self.cfg.max_depth,
                                              hole_val=self.cfg.hole_val)

        # Set subscribers and publishers
        if self.mode == 'sim':
            # Subscribers
            self.rgb_sub = rospy.Subscriber('/gibson_ros/camera/rgb/image', Image, self.rgb_cb)
            self.depth_sub = rospy.Subscriber('/gibson_ros/camera/depth/image', Image, self.depth_cb)
            if 'semantics' in ros_cfg['output']:  # If semantics are published
                self.sem_sub = rospy.Subscriber('/gibson_ros/camera/semantics/image', Image, self.sem_cb)
            self.odom_sub = rospy.Subscriber('/ground_truth_odom', Odometry, self.odom_cb)

            # Publishers
            self.vel_pub = rospy.Publisher('/navigation_velocity_smoother/raw_cmd_vel', Twist, queue_size=10)

            self.velocity_multiplier = 1.
        else:  # Real world
            # Subscribers
            self.depth_sub = rospy.Subscriber('/gibson_ros/camera_goggle/depth/image', Image, self.depth_cb)
            self.rgb_sub = rospy.Subscriber('/gibson_ros/camera_goggle/rgb/image', Image, self.rgb_cb)

            # Publishers
            self.vel_pub = rospy.Publisher('/cmd_vel_mux/gibson/raw_cmd_vel', Twist, queue_size=10)

            self.velocity_multiplier = 0.5  # Reduce speed of agent

        self.last_rgb = None
        self.last_depth = None
        # self.last_depth_raw = None
        self.last_sem = None

        # Ground truth odom
        self.last_position = None
        self.last_orientation = None

    def rgb_cb(self, image_message):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_message, desired_encoding="passthrough")
            self.last_rgb = cv_image
        except CvBridgeError as e:
            print(e)

    def depth_cb(self, image_message):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_message, desired_encoding="passthrough")
            self.last_depth = cv_image
        except CvBridgeError as e:
            print(e)

    def sem_cb(self, image_message):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_message, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)
        semantic_img = BagParser.get_semantic_idx(cv_image)
        self.last_sem = semantic_img

    def odom_cb(self, odom_message):
        self.last_position = np.array([odom_message.pose.pose.position.x,
                                  odom_message.pose.pose.position.y,
                                  odom_message.pose.pose.position.z,
                                  ]).astype(np.float32)
        self.last_orientation = np.array([odom_message.pose.pose.orientation.x,
                                     odom_message.pose.pose.orientation.y,
                                     odom_message.pose.pose.orientation.z,
                                     odom_message.pose.pose.orientation.w,
                                     ]).astype(np.float32)

    def create_vel_msg(self, vel):
        msg = Twist()
        msg.linear.x = vel[0]
        msg.linear.y = 0.
        msg.linear.z = 0.
        msg.angular.x = 0.
        msg.angular.y = 0.
        msg.angular.z = vel[1]
        return msg

    def execute_vel(self, output_vel):
        """Publish the velocity command.

        Args:
            output_vel: Velocity Torch tensor of shape (batch_size x 2) where batch_size should be 1.
        """
        output_vel = output_vel.cpu()
        output_vel = output_vel.data.numpy()
        output_vel = output_vel[0]  # Remove batch size dimension

        output_vel *= self.velocity_multiplier
        vel_msg = self.create_vel_msg(output_vel)

        self.vel_pub.publish(vel_msg)
