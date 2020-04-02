from __future__ import print_function

import glob
import numpy as np
import os
import rosbag
import rospy
import sys

from cv_bridge import CvBridge
from semnav.lib.image_map_transformer import Image2MapTransformer
from semnav.lib.utils import mkdir
from semnav.lib.categories import SemanticCategory


# Mapping according to: https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/core/channels/common/semantic_color.hpp#L15
rgb2semantic = {
    (0, 0, 0): SemanticCategory.UNK.value,
    (241, 255, 82): SemanticCategory.ceiling.value,
    (102, 168, 226): SemanticCategory.floor.value,
    (0, 255, 0): SemanticCategory.window.value,
    (113, 143, 65): SemanticCategory.door.value,
    (89, 173, 163): SemanticCategory.column.value,
    (254, 158, 137): SemanticCategory.beam.value,
    (190, 123, 75): SemanticCategory.wall.value,
    (100, 22, 116): SemanticCategory.sofa.value,
    (0, 18, 141): SemanticCategory.chair.value,
    (84, 84, 84): SemanticCategory.table.value,
    (85, 116, 127): SemanticCategory.board.value,
    (255, 31, 33): SemanticCategory.bookcase.value,
    (228, 228, 228): SemanticCategory.clutter.value,
    }


class BagParser(object):

    max_travel_dist = 0.4  # Meters
    time_between_frames = 0.2  # Seconds

    def __init__(self, bag_file, yaml_file, data_dir):
        if bag_file.endswith('.bag'):
            bag_files = [bag_file]
        else:
            bag_files = glob.glob(bag_file + '*.bag')
        self.bag_files = sorted(bag_files)
        self.data_dir = data_dir
        self.img2map = Image2MapTransformer(yaml_file)
        self.area_name = os.path.splitext(os.path.basename(yaml_file))[0]
        assert self.area_name.startswith('area_') is True

        self.bridge = CvBridge()
        self.idx_in_sequence = None
        self.sequence_idx = None
        self.last_room_name = 'UNK'

        # Make data directory if it doesn't exist
        mkdir(self.data_dir, verbose=True)

        self.use_semantics = True
        if self.use_semantics is True:
            raw_input('Using semantics. Continue if this is what you want (ctrl+C to quit)')
        else:
            raw_input('NOT using semantics. Continue if this is what you want (ctrl+C to quit)')

        self.topics = [
            '/navigation_velocity_smoother/raw_cmd_vel',
            '/gibson_ros/camera/depth/image',
            '/gibson_ros/camera/rgb/image',
            '/odom',
            '/ground_truth_odom',
            '/move_base/result',
            ]
        if self.use_semantics is True:
            self.topics.append('/gibson_ros/camera/semantics/image')

    @staticmethod
    def pose_to_numpy(pose):
        """Convert a Pose position to a numpy array [x, y, z].

        Args:
            pose: Pose position from ROS Pose message.

        Returns:
            pose_np: Numpy array of shape (3,) representing xyz pose.
        """
        return np.array([pose.x, pose.y, pose.z])

    @staticmethod
    def sem_rgb2idx(rgb_array):
        return rgb2semantic[tuple(rgb_array.tolist())]

    @staticmethod
    def get_semantic_idx_fast(semantics_rgb):
        """After measuring the timings, this is actually slower :(
        """
        semantics = np.apply_along_axis(BagParser.sem_rgb2idx, axis=2, arr=semantics_rgb)
        return semantics

    @staticmethod
    def get_semantic_idx(semantics_rgb):
        semantics = np.zeros((semantics_rgb.shape[0], semantics_rgb.shape[1]), dtype=np.uint8)
        for row in range(semantics_rgb.shape[0]):
            for col in range(semantics_rgb.shape[1]):
                rgb_tuple = tuple(semantics_rgb[row, col, :].tolist())
                category_index = rgb2semantic[rgb_tuple]
                semantics[row, col] = category_index
        return semantics

    def save_frame(self, frame, sequence_idx, idx_in_sequence):
        """Save a frame as an npz file.

        Data saved to npz file is a dict with the following keys:
            'vel': Numpy array of shape (2,) representing linear.x and angular.z. This is an
                    np.float64 converted to an np.float32.
            'rgb': Numpy float32 array of shape (H, W, C). Example: (240, 320, 3).
            'depth': Numpy float32 array of shape (H, W). Example: (240, 320). May contain nans.

        Args:
            frame: Dict containing frame data.
            sequence_idx: Index of the video sequence that this frame belongs to.
            idx_in_sequence: Index of current frame in the video sequence.
        """
        gt_odom = {
            'position': np.array([frame['gt_odom'].pose.pose.position.x,
                                  frame['gt_odom'].pose.pose.position.y,
                                  frame['gt_odom'].pose.pose.position.z,
                                  ]).astype(np.float32),
            'orientation': np.array([frame['gt_odom'].pose.pose.orientation.x,
                                     frame['gt_odom'].pose.pose.orientation.y,
                                     frame['gt_odom'].pose.pose.orientation.z,
                                     frame['gt_odom'].pose.pose.orientation.w,
                                     ]).astype(np.float32),
            }
        room_name = self.img2map.get_room_name(gt_odom['position'])

        # If room name is unknown, set to room name from previous frame
        if room_name is None:
            room_name = self.last_room_name

        # Update last room name
        self.last_room_name = room_name

        data = {
            'vel': np.array([frame['vel'].linear.x, frame['vel'].angular.z]).astype(np.float32),
            'rgb': np.asarray(self.bridge.imgmsg_to_cv2(frame['rgb'])),
            'depth': np.asarray(self.bridge.imgmsg_to_cv2(frame['depth'])),
            'gt_odom': gt_odom,
            'room_name': room_name,
            'sequence_idx': sequence_idx,  # Useful for debugging: checking our sequences are separated correctly
            'area_name': self.area_name,
            }
        if self.use_semantics is True:
            # Decode semantics
            data['semantics_rgb'] = np.asarray(self.bridge.imgmsg_to_cv2(frame['semantics'][0]))
            data['semantics'] = self.get_semantic_idx(data['semantics_rgb'])

        filepath = os.path.join(self.data_dir,
                                'sequence_%06d' % sequence_idx,
                                'traj_data_%06d.npz' % idx_in_sequence)
        np.savez(filepath, data)

    def mark_sequence_success(self, sequence_idx, status):
        """Mark whether the sequence was successful by creating a file.

        Args:
            sequence_idx: Sequence index.
            status: Status returned by /move_base/result.
        """
        assert status is not None

        filepath = os.path.join(self.data_dir,
                                'sequence_%06d' % sequence_idx,
                                'status.txt')
        assert os.path.isdir(os.path.dirname(filepath))
        assert os.path.isfile(filepath) is False
        with open(filepath, 'w') as f:
            if status == 3:
                f.write('success\n')
            else:
                f.write('failure\n')
            f.write('{}'.format(str(status) + '\n'))
        return sequence_idx + 1

    @classmethod
    def is_same_sequence(cls, frame, last_frame):
        if last_frame is None:
            return False  # It's actually better to return True here

        pose_begin = cls.pose_to_numpy(frame['odom'].pose.pose.position)
        pose_end = cls.pose_to_numpy(last_frame['odom'].pose.pose.position)

        travel_dist = np.linalg.norm(pose_begin - pose_end)

        if travel_dist > cls.max_travel_dist:
            return False
        else:
            return True

    def parse(self):
        sequence_idx = 0

        # Create an extra sequence_idx counter because it is not always in sync with sequence_idx
        result_sequence_idx = 1

        idx_in_sequence = 0

        last_odom = None
        last_gt_odom = None
        last_rgb = None
        last_depth = None
        if self.use_semantics is True:
            last_semantics = None
        last_vel = None
        last_frame = None

        time_window_end = None
        time_between_frames = rospy.Duration(self.time_between_frames)

        for bag_file in self.bag_files:
            print('Parsing bag file:', bag_file)
            bag = rosbag.Bag(bag_file)
            for topic, msg, cur_msg_time in bag.read_messages():
                if (time_window_end is None) or (cur_msg_time > time_window_end):
                    # Update time_window_end
                    if time_window_end is None:
                        time_window_end = cur_msg_time
                    time_window_end += time_between_frames

                    if not ((last_odom is None)
                            or (last_gt_odom is None)
                            or (last_rgb is None)
                            or (last_depth is None)
                            or (last_vel is None)):
                        if (self.use_semantics is False) or ((self.use_semantics is True) and (last_semantics is not None)):

                            # New frame
                            frame = {
                                'odom': last_odom,
                                'gt_odom': last_gt_odom,
                                'depth': last_depth,
                                'rgb': last_rgb,
                                'vel': last_vel,
                                }
                            if self.use_semantics is True:
                                frame['semantics'] = last_semantics,

                            # Update sequence index if necessary
                            if not self.is_same_sequence(frame, last_frame):
                                sequence_idx += 1
                                idx_in_sequence = 0
                                sequence_dir = os.path.join(self.data_dir,
                                                            'sequence_%06d' % sequence_idx)
                                mkdir(sequence_dir, verbose=False)
                                self.last_room_name = 'UNK'
                            else:  # Ignore first frame of each sequence
                                # Save new frame
                                self.save_frame(frame, sequence_idx, idx_in_sequence)

                            # Update last frame and increment counters
                            last_frame = frame
                            idx_in_sequence += 1
                if topic in self.topics:

                    if topic == '/odom':
                        last_odom = msg
                    elif topic == '/ground_truth_odom':
                        last_gt_odom = msg
                    elif topic == '/gibson_ros/camera/depth/image':
                        last_depth = msg
                    elif topic == '/gibson_ros/camera/rgb/image':
                        last_rgb = msg
                    elif topic == '/gibson_ros/camera/semantics/image':
                        last_semantics = msg
                    elif topic == '/move_base/result':
                        pass
                        result_sequence_idx += 1
                    elif topic == '/navigation_velocity_smoother/raw_cmd_vel':
                        last_vel = msg
            bag.close()


def main():
    bag_file = sys.argv[1]
    yaml_file = sys.argv[2]
    prefix = sys.argv[3]

    print('Bag file (or bag file prefix):', bag_file)
    print('Yaml file:', yaml_file)
    print('Destination:', prefix)
    print('')

    bag_parser = BagParser(bag_file, yaml_file, prefix)
    bag_parser.parse()


if __name__ == '__main__':
    main()
