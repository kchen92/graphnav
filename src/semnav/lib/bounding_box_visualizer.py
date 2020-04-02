#!/usr/bin/env python

"""Visualize bounding boxes (which are identified by name).
"""

import numpy as np
import rospy

from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray


class BoundingBoxVisualizer(object):
    """A node that publishes a BoundingBoxArray().

    A bounding box is a dict containing the following keys:
        - upper_left_px: [row, col] of upper left bounding box corner in pixel coordinates
        - lower_right_px: [row, col] of lower right bounding box corner in pixel coordinates
    """

    def __init__(self, img2map, topic=None):
        self.img2map = img2map
        self.bb_msg_list_dict = {}
        if topic is None:
            topic = '/bounding_box_array'
        self.bb_pub = rospy.Publisher(topic, BoundingBoxArray, queue_size=10)

        self.bb_array_msg = self.create_bb_array_msg([])

        # self.bb_pub.publish(self.bb_array_msg)
        # rospy.spin()

        # rate = rospy.Rate(10) # 10hz
        # while not rospy.is_shutdown():
        #     self.bb_pub.publish(self.bb_array_msg)
        #     rate.sleep()

    def clear(self):
        self.bb_msg_list_dict = {}

    def create_bb_msg(self, cur_bb, is_map_coord=False):
        bb_corners = np.array([cur_bb['upper_left_px'], cur_bb['lower_right_px']])
        if is_map_coord is True:
            bb_corners_map_coord = bb_corners
        else:
            bb_corners_px_coord = bb_corners
            bb_corners_map_coord = self.img2map.pixel2map(bb_corners_px_coord)
        bb_corners_mean = np.mean(bb_corners_map_coord, axis=0)
        bb_corners_diff = np.abs(bb_corners_map_coord[0, :] - bb_corners_map_coord[1, :])

        cur_bb_msg = BoundingBox()
        cur_bb_msg.header.frame_id = 'map'
        cur_bb_msg.header.stamp = rospy.Time.now()
        cur_bb_msg.pose.position.x = bb_corners_mean[0]
        cur_bb_msg.pose.position.y = bb_corners_mean[1]
        cur_bb_msg.pose.position.z = 0
        cur_bb_msg.pose.orientation.x = 0
        cur_bb_msg.pose.orientation.y = 0
        cur_bb_msg.pose.orientation.z = 0
        cur_bb_msg.pose.orientation.w = 1
        cur_bb_msg.dimensions.x = bb_corners_diff[0]
        cur_bb_msg.dimensions.y = bb_corners_diff[1]
        cur_bb_msg.dimensions.z = 0.1
        cur_bb_msg.value = 1.0
        cur_bb_msg.label = 1
        return cur_bb_msg

    def create_bb_msg_list(self, names, bounding_box_list, is_map_coord=False):
        """Create a dict that maps name -> BoundingBox msg.

        Args:
            names: List of bounding box names.
            bounding_box_list: List of bounding boxes (defined in class name).

        Returns:
            boxes: A dict mapping from name -> BoundingBox msg.
        """
        # NOTE: The conversion from pixel to map coordinates can be vectorized across the bounding box
        # list if necessary.
        boxes = {}
        for cur_name, cur_bb in zip(names, bounding_box_list):
            cur_bb_msg = self.create_bb_msg(cur_bb, is_map_coord=is_map_coord)
            boxes[cur_name] = cur_bb_msg
        return boxes

    def create_bb_array_msg(self, bb_msg_list):
        """Create a BoundingBoxArray msg from a bounding box list dict (output from
        self.create_bb_msg_list).

        Args:
            bounding_box_list: A list of bounding box dicts as defined in read_bounding_boxes().

        Returns:
            oh_bb: An instance of the BoundingBoxArray msg.
        """
        oh_bb = BoundingBoxArray()
        oh_bb.header.frame_id = 'map'
        oh_bb.header.stamp = rospy.Time.now()
        oh_bb.boxes = bb_msg_list
        return oh_bb

    def add_bounding_box_list(self, names, bounding_box_list, is_map_coord=False):
        # Create a list of bounding box messages for new bounding box list
        new_bb_msg_list_dict = self.create_bb_msg_list(names, bounding_box_list,
                                                       is_map_coord=is_map_coord)

        # Add it to the existing bounding box list
        for cur_name, cur_bb in new_bb_msg_list_dict.iteritems():
            if cur_name in self.bb_msg_list_dict:
                raise ValueError('Name already exists in bounding box list!')
            self.bb_msg_list_dict[cur_name] = cur_bb

        # Create the new BoundingBoxArray message
        self.bb_array_msg = self.create_bb_array_msg(self.bb_msg_list_dict.values())

        # Publish updated BoundingBoxArray message
        self.publish_bb()

    def add_bounding_box(self, name, bounding_box):
        self.add_bounding_box_list([name], [bounding_box])

    def publish_bb(self):
        self.bb_pub.publish(self.bb_array_msg)
