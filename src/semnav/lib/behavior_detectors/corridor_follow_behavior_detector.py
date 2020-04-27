from semnav.lib.behavior_detectors import BehaviorDetector
from semnav.lib.utils import get_frame_paths, get_frame_idx, load_frame


class CorridorFollowBehaviorDetector(BehaviorDetector):
    """Corridor follow behavior detector. This behavior detector assumes it is run last among the
    different behavior detectors. Any frame within a corridor that is not labeled as another
    behavior is considered a corridor follow behavior.
    """

    def __init__(self):
        super(CorridorFollowBehaviorDetector, self).__init__()
        self.rooms = {  # Overwrite self.rooms from base class
            'hallway',
            'openspace',
            'lounge',
            }

    def process_sequence(self, sequence_dir):
        frame_paths = get_frame_paths(sequence_dir)
        for cur_frame_path in frame_paths:
            cur_frame = load_frame(cur_frame_path)
            is_cf = (cur_frame.get('behavior_id') is None) and self.in_room(cur_frame)
            if is_cf:
                cur_frame_idx = get_frame_idx(cur_frame_path)
                self.add_behavior_label(sequence_dir, cur_frame_idx, behavior_id='cf')
