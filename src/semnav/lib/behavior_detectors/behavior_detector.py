import numpy as np
import os

from semnav.lib.utils import get_frame_paths, load_frame


class BehaviorDetector(object):

    def __init__(self):
        """Constructor.

        Args:
            turn_yaw_threshold: Threshold (in radians) for detecting when a turn has occurred.
        """
        self.rooms = {  # Types of rooms that are considered to be rooms (vs. hallway)
            'office',
            'conferenceRoom',
            'copyRoom',
            'pantry',
            'WC',
            }

    @staticmethod
    def get_frame_path_at_idx(sequence_dir, idx):
        """Get the (full) path to the frame with the specified index (1-based indexing).
        """
        return os.path.join(sequence_dir, 'traj_data_%06d.npz' % idx)

    def in_room_by_name(self, room_name):
        return any([room_name.startswith(cur_room_type) for cur_room_type in self.rooms])

    def in_room(self, cur_frame):
        """Return whether the agent is in a room at the current frame.
        """
        return self.in_room_by_name(cur_frame['room_name'])

    def is_transition_point(self):
        raise NotImplementedError('Must be implemented by subclass.')

    def find_transition_points(self, sequence_dir):
        """Finds the transition points and assembles them in a list.
        """
        frame_paths = get_frame_paths(sequence_dir)
        last_frame = None
        last_frame_path = None
        transition_points = []
        for cur_frame_path in frame_paths:
            cur_frame = load_frame(cur_frame_path)
            if self.is_transition_point(last_frame, cur_frame):
                transition_point = {'begin': last_frame_path,
                                    'end': cur_frame_path}
                transition_points.append(transition_point)

            # Update last frame
            last_frame = cur_frame
            last_frame_path = cur_frame_path

        return transition_points

    def add_behavior_label(self, sequence_dir, cur_frame_idx, behavior_id):
        cur_frame_path = self.get_frame_path_at_idx(sequence_dir,
                                                    cur_frame_idx)
        cur_frame = load_frame(cur_frame_path)
        cur_frame['behavior_id'] = behavior_id  # Overwrite behavior ID
        np.savez(cur_frame_path, cur_frame)  # Overwrite the .npz file

    def add_behavior_labels(self, sequence_dir, start_frame_idx, end_frame_idx, behavior_id):
        """Add behavior labels by overwriting the npz data according to the specified frame indices
        (inclusive).

        Args:
            sequence_dir: Full path to the sequence directory.
            start_frame_idx: 1-based index of the start frame of the behavior.
            end_frame_idx: 1-based index of the end frame of the behavior.
            behavior_id: Behavior ID (e.g. 'oor').
        """
        assert end_frame_idx >= start_frame_idx
        for cur_frame_idx in range(start_frame_idx, end_frame_idx + 1):
            self.add_behavior_label(sequence_dir, cur_frame_idx, behavior_id)

    def process_transition_point(self):
        raise NotImplementedError('Must be implemented by subclass.')

    def process_sequence(self, sequence_dir):
        transition_points = self.find_transition_points(sequence_dir)
        for transition_point in transition_points:
            start, end, behavior_id = self.process_transition_point(sequence_dir, transition_point)
            if behavior_id is not None:
                self.add_behavior_labels(sequence_dir, start, end, behavior_id)
