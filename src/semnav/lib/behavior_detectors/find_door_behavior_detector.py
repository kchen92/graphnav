from semnav.lib.behavior_detectors import BehaviorDetector
from semnav.lib.utils import get_frame_idx


class FindDoorBehaviorDetector(BehaviorDetector):

    def __init__(self):
        super(FindDoorBehaviorDetector, self).__init__()

    def is_transition_point(self, last_frame, cur_frame):
        """Determine whether this is a transition point.
        """
        if last_frame is None:
            return False
        else:
            return self.in_room(last_frame) and cur_frame['room_name'].startswith('hallway')

    def find_transition_points(self, sequence_dir):
        """Finds the transition points and assembles them in a list.
        """
        transition_points = super(FindDoorBehaviorDetector, self).find_transition_points(sequence_dir)
        assert len(transition_points) <= 1  # Agent only exits room once in each sequence
        return transition_points

    def get_behavior(self, transition_yaw, cur_yaw, forward_time):
        raise NotImplementedError('We do not need this for the FindDoorBehaviorDetector.')

    def find_behavior_start(self, sequence_dir, transition_point):
        """Find the frame at which the behavior begins.
        Args:
            sequence_dir: Full path to the sequence directory.
            transition point: Dict representing transition point with 'begin' and 'end' frames.
        """
        # Assume that the trajectory begins inside the office
        transition_frame_idx = 1  # Frames are 1-indexed
        return transition_frame_idx

    def find_behavior_end(self, sequence_dir, transition_point):
        """Find the frame at which the behavior ends given the transition point.

        Args:
            sequence_dir: Full path to the sequence directory.
            transition_point: Dict representing transition point with 'begin' and 'end' frames.

        Returns:
            end_frame_idx: 1-based index of the end behavior frame in the sequence.
            behavior_id: Behavior ID (e.g. 'oor').
        """
        transition_frame_path = transition_point['begin']
        transition_frame_idx = get_frame_idx(transition_frame_path)
        # Return transition_frame_idx because frames close to door will get overwritten anyway
        return transition_frame_idx

    def process_transition_point(self, sequence_dir, transition_point):
        start = self.find_behavior_start(sequence_dir, transition_point)
        end = self.find_behavior_end(sequence_dir, transition_point)
        behavior_id = 'fd'
        return start, end, behavior_id
