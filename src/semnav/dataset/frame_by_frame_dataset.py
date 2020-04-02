"""PyTorch Dataset where each sample is one timestep frame.
"""

import numpy as np
import torch

from semnav.dataset.trajectory_dataset import TrajectoryDataset


class FrameByFrameDataset(TrajectoryDataset):

    def __init__(self, root_dir, first_n_in_sequence=None, remove_last_n_in_sequence=None,
                 behavior_ids=None, valid_only=True):
        """Constructor.

        Args:
            first_n_in_sequence: This dataset will only use the first first_n_in_sequence frames of
                    each sequence. All other frames in each sequence will be discarded.
            remove_last_n_in_sequence: If there are 40 frames and remove_last_n_in_sequence == 1,
                    then only the very last frame will be removed.
        """
        super(FrameByFrameDataset, self).__init__(root_dir, first_n_in_sequence=first_n_in_sequence,
                                                  remove_last_n_in_sequence=remove_last_n_in_sequence,
                                                  behavior_ids=behavior_ids, valid_only=valid_only)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        return self.get_frame_data(self.filenames[idx])
