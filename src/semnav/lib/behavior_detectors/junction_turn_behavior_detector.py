from __future__ import division

import numpy as np
import os

from semnav.lib.behavior_detectors import BehaviorDetector
from semnav.lib.room_border_finder import RoomBorderFinder, RoomAdjRelationship
from semnav.lib.utils import compute_angle_delta, get_frame_idx, get_frame_paths, load_frame
try:
    from tf.transformations import euler_from_quaternion
except ImportError:
    from semnav.third_party.transformations import euler_from_quaternion


class JunctionTurnBehaviorDetector(BehaviorDetector):

    def __init__(self, area_yaml, max_turn_dist=2., turn_yaw_threshold=(np.pi / 4.5),
                 n_after_turn_frames=20, adjacent_edge_threshold=1.5):
        """Constructor.

        Args:
            area_yaml: Filepath to yaml for the current area.
            max_turn_dist: Distance (meters) beyond a transition point to search for an orientation
                change.
            turn_yaw_threshold: Threshold (in radians) for detecting when a turn has occurred.
            n_after_turn_frames: Number of frames to also add on as part of the turn behavior after
                a turn has been detected.
            adjacent_edge_threshold: Max distance (meters) between two bounding box edges from two
                different rooms for them to be considered adjacent.
        """
        super(JunctionTurnBehaviorDetector, self).__init__()
        self.max_turn_dist = max_turn_dist
        self.turn_yaw_threshold = turn_yaw_threshold
        self.n_after_turn_frames = n_after_turn_frames
        self.room_border_finder = RoomBorderFinder(yaml_filepath=area_yaml,
                                                   adjacent_edge_threshold=adjacent_edge_threshold)

        # Set of valid rooms for junctions
        self.junction_rooms = [
            'hallway',
            'lobby',
            'lounge',
            'openspace',
            ]
        self.junction_rooms.extend(self.rooms)  # Room intersections count as junctions

    def is_transition_point(self, last_frame, cur_frame):
        """Determine whether this is a transition point.
        """
        if last_frame is None:
            return False
        elif last_frame['room_name'] != cur_frame['room_name']:  # Transitioning rooms
            return ((last_frame['room_name'].split('_')[0] in self.junction_rooms)
                    and (cur_frame['room_name'].split('_')[0] in self.junction_rooms))
        else:
            return False

    def get_behavior(self, transition_yaw, cur_yaw, forward_time):
        """Return the behavior ID corresponding to the given transition point.
        """
        if forward_time is True:
            yaw_delta = compute_angle_delta(transition_yaw, cur_yaw)
            if yaw_delta > self.turn_yaw_threshold:
                return 'tl'
            elif -yaw_delta > self.turn_yaw_threshold:
                return 'tr'
        else:
            yaw_delta = compute_angle_delta(cur_yaw, transition_yaw)
            if yaw_delta > self.turn_yaw_threshold:
                return 'tl'
            elif -yaw_delta > self.turn_yaw_threshold:
                return 'tr'
        return None

    def get_transition_yaw(self, begin_frame, end_frame):
        row_border, col_border, relationship = self.room_border_finder.get_border(
            begin_frame['room_name'], end_frame['room_name'])

        if (row_border is None) and (col_border is None):
            raise ValueError('Rooms should be adjacent, but this was not detected.')

        # Determine yaw orientation
        if relationship is RoomAdjRelationship.LEFT:
            # print('Going from left to right')
            transition_yaw = 0.
        elif relationship is RoomAdjRelationship.RIGHT:
            # print('Going from right to left')
            transition_yaw = np.pi
        elif relationship is RoomAdjRelationship.ABOVE:
            # print('Going from top to bottom')
            transition_yaw = 3. * np.pi / 2
        elif relationship is RoomAdjRelationship.BELOW:
            # print('Going from bottom to top')
            transition_yaw = np.pi / 2
        else:
            raise ValueError
        return transition_yaw

    def find_behavior_start(self, sequence_dir, transition_point):
        """Find the frame at which the behavior begins.

        Args:
            sequence_dir: Full path to the sequence directory.
            transition point: Dict representing transition point with 'begin' and 'end' frames.
        """
        transition_frame_path = transition_point['end']
        begin_frame = load_frame(transition_point['begin'])
        end_frame = load_frame(transition_point['end'])
        transition_yaw = self.get_transition_yaw(begin_frame, end_frame)

        transition_frame_path = {'transition_yaw': transition_yaw,
                                 'transition_frame_path': transition_frame_path}

        cur_frame_idx, behavior_id = self.find_turn(sequence_dir, transition_frame_path,
                                                    forward_time=False)
        if behavior_id is not None:
            cur_frame_idx = max(cur_frame_idx - self.n_after_turn_frames, 1)
        return cur_frame_idx, behavior_id

    def find_behavior_end(self, sequence_dir, transition_point):
        """Find the frame at which the behavior ends given the transition point.

        Args:
            sequence_dir: Full path to the sequence directory.
            transition_point: Dict representing transition point with 'begin' and 'end' frames.

        Returns:
            end_frame_idx: 1-based index of the end behavior frame in the sequence.
            behavior_id: Behavior ID (e.g. 'oor').
        """
        transition_frame_path = transition_point['end']
        begin_frame = load_frame(transition_point['begin'])
        end_frame = load_frame(transition_point['end'])

        if not self.is_enter_room(begin_frame, end_frame):
            transition_yaw = self.get_transition_yaw(begin_frame, end_frame)

            transition_frame_path = {'transition_yaw': transition_yaw,
                                     'transition_frame_path': transition_frame_path}

            cur_frame_idx, behavior_id = self.find_turn(sequence_dir, transition_frame_path,
                                                        forward_time=True)
        else:  # Special case for when entering a room, since turning after entering does not count
            cur_frame_idx = get_frame_idx(transition_frame_path)
            behavior_id = None
        if behavior_id is not None:
            last_frame_idx = len(get_frame_paths(sequence_dir))
            cur_frame_idx = min(cur_frame_idx + self.n_after_turn_frames, last_frame_idx)

        return cur_frame_idx, behavior_id

    def process_transition_point(self, sequence_dir, transition_point):
        start, start_behavior_id = self.find_behavior_start(sequence_dir, transition_point)
        end, end_behavior_id = self.find_behavior_end(sequence_dir, transition_point)

        # Begin frame (frame at transition point)
        begin_frame = load_frame(transition_point['begin'])

        # Start frame
        start_frame_path = self.get_frame_path_at_idx(sequence_dir, start)
        start_frame = load_frame(start_frame_path)
        if (start_behavior_id is not None) and (end_behavior_id is not None):
            # If the behaviors are different, "smartly" choose a behavior by seeing which one
            # (start vs. end) has greater yaw delta with transition point
            # However, if exiting a room, choose the end behavior instead of start

            # Start frame orientation
            start_quaternion = start_frame['gt_odom']['orientation']
            start_yaw = euler_from_quaternion(start_quaternion)[2]

            # End frame orientation
            end_frame_path = self.get_frame_path_at_idx(sequence_dir, end)
            end_frame = load_frame(end_frame_path)
            end_quaternion = end_frame['gt_odom']['orientation']
            end_yaw = euler_from_quaternion(end_quaternion)[2]

            if start_frame['room_name'] == end_frame['room_name']:
                # This is for an edge case where the robot goes from room A to room B to room A,
                # after spending like 0.5 seconds in room B. In this case, the start and end frames
                # are both in room A and the rooms are not adjacent, failing self.get_transition_yaw
                print('Rooms are not adjacent!')
                return start, end, None
            else:
                transition_yaw = self.get_transition_yaw(start_frame, end_frame)

            # Choose a behavior
            start_delta = np.abs(compute_angle_delta(start_yaw, transition_yaw))
            end_delta = np.abs(compute_angle_delta(end_yaw, transition_yaw))
            if start_delta > end_delta:
                behavior_id = start_behavior_id
            else:
                behavior_id = end_behavior_id

        if (start_behavior_id is None) and (end_behavior_id is None):
            # Determine whether robot is going straight into a room or not
            end_frame_path = self.get_frame_path_at_idx(sequence_dir, end)
            end_frame = load_frame(end_frame_path)
            if begin_frame['room_name'].startswith('hallway') and self.in_room(end_frame):
                behavior_id = 's'
            else:
                # Let CorridorFollowBehaviorDetector take care of cf
                # We don't want to overwrite previous 'tr' frames by accident
                behavior_id = None
                # behavior_id = 'cf'
        if (start_behavior_id is not None) and (end_behavior_id is None):
            behavior_id = start_behavior_id
        elif (end_behavior_id is not None) and (start_behavior_id is None):
            behavior_id = end_behavior_id

        # If start frame is in room, choose end_behavior_id
        if self.in_room(begin_frame):
            if end_behavior_id is None:
                behavior_id = 'cf'
            else:
                behavior_id = end_behavior_id
        return start, end, behavior_id

    @staticmethod
    def _is_enter_room(last_frame, cur_frame, room_name):
        """Determine whether the agent is entering a room with the specific room name.
        """
        return (not last_frame['room_name'].startswith(room_name)
                and cur_frame['room_name'].startswith(room_name))

    def is_enter_room(self, last_frame, cur_frame):
        """Determine whether this is an enter room transition point.
        """
        if last_frame is None:
            return False
        else:
            return any([self._is_enter_room(last_frame, cur_frame, cur_room_type)
                        for cur_room_type in self.rooms])

    def in_turn_loop(self, begin_frame, cur_frame):
        return cur_frame['room_name'] == begin_frame['room_name']

    def find_turn(self, sequence_dir, transition_frame_path, forward_time):
        if not isinstance(transition_frame_path, str):  # JunctionTurnBehaviorDetector special case
            input_yaw = transition_frame_path['transition_yaw']
            transition_frame_path = transition_frame_path['transition_frame_path']
            use_input_yaw = True
        else:
            use_input_yaw = False
        transition_frame = load_frame(transition_frame_path)
        transition_quaternion = transition_frame['gt_odom']['orientation']
        if use_input_yaw is True:
            transition_yaw = input_yaw
        else:
            transition_yaw = euler_from_quaternion(transition_quaternion)[2]
        transition_pos = transition_frame['gt_odom']['position']
        cur_pos = transition_pos

        transition_frame_idx = get_frame_idx(transition_frame_path)

        # Frame at the start of find_turn
        if forward_time is True:
            begin_frame = transition_frame
        else:
            begin_frame_path = self.get_frame_path_at_idx(sequence_dir, transition_frame_idx - 1)
            begin_frame = load_frame(begin_frame_path)

        cur_frame_idx = transition_frame_idx
        frame_delta = 1
        in_turn_loop = True
        while (in_turn_loop
               and (np.linalg.norm(cur_pos - transition_pos) < self.max_turn_dist)):
            prev_frame_idx = cur_frame_idx
            if forward_time is True:  # Move forward in time
                cur_frame_idx = transition_frame_idx + frame_delta
            else:  # Move backwards in time
                cur_frame_idx = transition_frame_idx - frame_delta
            cur_frame_path = self.get_frame_path_at_idx(sequence_dir,
                                                        cur_frame_idx)
            # Break if we have reached beginning/end of sequence
            if not os.path.isfile(cur_frame_path):
                cur_frame_idx = prev_frame_idx
                break
            cur_frame = load_frame(cur_frame_path)
            cur_quaternion = cur_frame['gt_odom']['orientation']
            cur_yaw = euler_from_quaternion(cur_quaternion)[2]
            cur_pos = cur_frame['gt_odom']['position']

            behavior_id = self.get_behavior(transition_yaw, cur_yaw, forward_time)
            if behavior_id is not None:
                return cur_frame_idx, behavior_id

            # Update counters
            in_turn_loop = self.in_turn_loop(begin_frame, cur_frame)
            frame_delta += 1
        return cur_frame_idx, None
