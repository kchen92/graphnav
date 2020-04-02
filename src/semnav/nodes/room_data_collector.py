#!/usr/bin/env python

"""Data collection for going from room to room.
"""

from __future__ import print_function
from __future__ import division

import argparse
import numpy as np
import os
import random
import rospy
import yaml

from geometry_msgs.msg import PoseStamped
from semnav.lib.room_sampler import RoomSampler
from semnav.lib.discrete_sampler import DiscreteSampler
from semnav.lib.image_map_transformer import Image2MapTransformer
from semnav.lib.robot_navigator import RobotNavigator
from semnav.lib.bounding_box_visualizer import BoundingBoxVisualizer
from semnav.lib.utils import compute_orientation, dilate_map, read_area_data
from tf.transformations import quaternion_from_euler


class RoomDataCollector(object):

    UNIFORM_SAMPLING = False

    def __init__(self, start_navigator=True, sample_rooms_only=False):
        self.sample_rooms_only = sample_rooms_only
        if self.sample_rooms_only is True:
            print('Sampling from ROOMS ONLY')
        else:
            print('Sampling from ALL ROOMS')

        rospy.init_node('room_data_collector')
        yaml_filepath = rospy.get_param('map_file')
        print(yaml_filepath)

        # Load yaml file containing map data
        with open(yaml_filepath, 'r') as f:
            map_yaml = yaml.load(f)

        self.img2map = Image2MapTransformer(yaml_filepath)

        mat_filepath = os.path.join(os.path.dirname(yaml_filepath), map_yaml['mat_filepath'])
        self.area_data = read_area_data(mat_filepath)
        if start_navigator is True:  # This is False when running NavPlanFileCreator
            self.start_navigator()

        # Find valid rooms
        self.valid_rooms = []
        rooms = self.area_data['rooms']
        for room in rooms:
            if ((room['name'] not in map_yaml.get('invalid_rooms', []))
                    and self.is_correct_room_type(room)):
                # Shrink the clutter map dilation kernel size until we find a valid kernel size
                kernel_size = 11
                found_valid_kernel = False
                while not found_valid_kernel:
                    if kernel_size < 1:
                        raise ValueError('Cannot find good endpoints for room')
                    try:
                        # Dilate clutter map
                        # Dilation kernel shape should depend on map resolution
                        dilated_clutter_map = dilate_map(self.area_data['clutter_map'],
                                                         kernel_shape=[kernel_size, kernel_size])

                        # Make a room sampler
                        room_sampler = self.create_room_sampler(dilated_clutter_map, room)
                    except AssertionError:
                        kernel_size -= 2
                        continue
                    found_valid_kernel = True
                if room_sampler.is_valid:
                    self.valid_rooms.append(room_sampler)

        # Bounding box visualization
        self.bb_visualizer = BoundingBoxVisualizer(self.img2map)
        rospy.sleep(1.)  # Give time to establish connections
        print('Number of valid rooms:', len(self.valid_rooms))
        names = []
        bounding_boxes = []
        for cs in self.valid_rooms:
            names.append(cs.name)
            bounding_boxes.append(cs.bounding_box)
        self.bb_visualizer.add_bounding_box_list(names, bounding_boxes)

        if not self.UNIFORM_SAMPLING:
            # Build a sampler for sampling which room to choose based on bounding box area
            room_areas = []
            for cur_room in self.valid_rooms:
                cur_area = cur_room.long_edge * cur_room.short_edge
                room_areas.append(cur_area)
            self.room_sampler = DiscreteSampler(self.valid_rooms, room_areas)

        self.episode_count = 0

        # Robot teleport publisher (reset pose)
        self.tp_pub = rospy.Publisher('reset_pose', PoseStamped, queue_size=10)
        rospy.sleep(3.)  # Wait for connections to be established

    def run(self):
        n_success = 0
        while not rospy.is_shutdown():
            print('Starting new episode...')

            self.episode_count += 1

            # Publish bounding box visualizer
            self.bb_visualizer.publish_bb()

            # Sample a trajectory
            print('Sampling trajectory...')
            cur_traj = self.sample_trajectory()

            # Convert from pixel coordinates to xy coordinates
            start_pt_xy = self.img2map.pixel2map(cur_traj['start_pt'])
            end_pt_xy = self.img2map.pixel2map(cur_traj['end_pt'])

            # Compute robot orientation
            print('Computing initial orientation')
            start_orientation = self.compute_orientation(cur_traj)

            # Teleport robot
            print('Teleporting robot')
            tp_msg = self.get_tp_msg(start_pt_xy, start_orientation)
            self.tp_pub.publish(tp_msg)

            # End orientation
            end_orientation = compute_orientation(cur_traj['start_pt'], cur_traj['end_pt'],
                                                  noise=0)  # Euler

            # Give it time to reposition
            rospy.sleep(0.5)  # This isnt super necessary

            # Send the nav goal
            print('Sending robot goal...')
            end_pt_z = 0.
            end_pt = np.array([end_pt_xy[0], end_pt_xy[1], end_pt_z])
            success = self.send_goal(end_pt, end_orientation)

            # Update statistics
            if success:
                n_success += 1

    def compute_orientation(self, cur_traj):
        start_orientation = compute_orientation(cur_traj['start_pt'], cur_traj['end_pt'],
                                                noise=60)  # Euler
        return start_orientation

    def start_navigator(self):
        self.robot_navigator = RobotNavigator(move_base_timeout=300, pause_interval=2)

    def send_goal(self, end_pt, end_orientation):
        robot_stuck = self.robot_navigator.set_goal(end_pt[0], end_pt[1], end_pt[2],
                                                    end_orientation)
        return not robot_stuck

    def is_correct_room_type(self, room):
        """Check if the room type is correct.
        """
        if self.sample_rooms_only:
            return (not room['name'].startswith('hallway')
                    and not room['name'].startswith('storage')
                    and not room['name'].startswith('openspace')
                    and not room['name'].startswith('lounge'))
        else:
            return True

    def create_room_sampler(self, dilated_clutter_map, room):
        return RoomSampler(self.area_data['semantic_map'],
                           dilated_clutter_map,
                           room,
                           self.area_data['resolution'])

    def sample_trajectory(self):
        """Sample a trajectory.

        Returns:
            cur_traj: Dict containing start_pt and end_pt in pixel coordinates.
        """
        start_pt = None
        end_pt = None
        while (start_pt is None) or (end_pt is None):
            while True:
                start_room_sampler = self.sample_room()  # Select a room to sample from
                end_room_sampler = self.sample_room()  # Select a room to sample from
                if start_room_sampler is not end_room_sampler:
                    break

            start_pt = start_room_sampler.sample_once()
            end_pt = end_room_sampler.sample_once()
        print('Start room:', start_room_sampler.name)
        print('End room:', end_room_sampler.name)
        cur_traj = {
            'start_pt': start_pt,
            'end_pt': end_pt,
            'start_room': start_room_sampler.name,
            'end_room': end_room_sampler.name,
            }
        return cur_traj

    def get_tp_msg(self, start_xy, theta):
        tp_msg = PoseStamped()
        tp_msg.header.frame_id = 'map'
        tp_msg.header.stamp = rospy.Time.now()
        tp_msg.pose.position.x = start_xy[0]
        tp_msg.pose.position.y = start_xy[1]
        tp_msg.pose.position.z = 0.5

        # Compute orientation from theta
        quat = quaternion_from_euler(0, 0, theta)
        tp_msg.pose.orientation.x = quat[0]
        tp_msg.pose.orientation.y = quat[1]
        tp_msg.pose.orientation.z = quat[2]
        tp_msg.pose.orientation.w = quat[3]
        return tp_msg

    def sample_room(self):
        """Choose a room sampler from self.valid_rooms.
        """
        if self.UNIFORM_SAMPLING:
            return random.choice(self.valid_rooms)  # Uniform sampling
        else:
            return self.room_sampler.sample_once()  # Sample based on area


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Room data collector')
    parser.add_argument('--sample_rooms_only', action='store_true')
    args = parser.parse_args()

    try:
        room_data_collector = RoomDataCollector(sample_rooms_only=args.sample_rooms_only)
        room_data_collector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Room data collector terminated.")
