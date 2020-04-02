from __future__ import print_function

import os
import pickle
import random

from semnav.dataset.dataset_utils import prepend_dataset_root
from semnav.dataset.trajectory_dataset import TrajectoryDataset


class TemporalDataset(TrajectoryDataset):
    """Dataset which returns a sequence of frames for each sample.

    Example:
        temporal_dilation=3
        n_frames_per_sample=10

        Index 0 of dataset:
            <path_to_data>/sequence_000013/traj_data_000001.npz
            <path_to_data>/sequence_000013/traj_data_000004.npz
            <path_to_data>/sequence_000013/traj_data_000007.npz
            <path_to_data>/sequence_000013/traj_data_000010.npz
            <path_to_data>/sequence_000013/traj_data_000013.npz
            <path_to_data>/sequence_000013/traj_data_000016.npz
            <path_to_data>/sequence_000013/traj_data_000019.npz
            <path_to_data>/sequence_000013/traj_data_000022.npz
            <path_to_data>/sequence_000013/traj_data_000025.npz
            <path_to_data>/sequence_000013/traj_data_000028.npz

        Index 1 of dataset:
            <path_to_data>/sequence_000013/traj_data_000002.npz
            <path_to_data>/sequence_000013/traj_data_000005.npz
            <path_to_data>/sequence_000013/traj_data_000008.npz
            <path_to_data>/sequence_000013/traj_data_000011.npz
            <path_to_data>/sequence_000013/traj_data_000014.npz
            <path_to_data>/sequence_000013/traj_data_000017.npz
            <path_to_data>/sequence_000013/traj_data_000020.npz
            <path_to_data>/sequence_000013/traj_data_000023.npz
            <path_to_data>/sequence_000013/traj_data_000026.npz
            <path_to_data>/sequence_000013/traj_data_000029.npz
    """

    def __init__(self, root_dir, temporal_dilation=1, n_frames_per_sample=5,
                 first_n_in_sequence=None, remove_last_n_in_sequence=None, behavior_ids=None,
                 valid_only=True):
        """Constructor.

        Args:
            first_n_in_sequence: This dataset will only use the first first_n_in_sequence frames of
                    each sequence. All other frames in each sequence will be discarded.
            remove_last_n_in_sequence: If there are 40 frames and remove_last_n_in_sequence == 1,
                    then only the very last frame will be removed.
        """
        assert temporal_dilation is not None
        assert n_frames_per_sample is not None
        super(TemporalDataset, self).__init__(root_dir, first_n_in_sequence=first_n_in_sequence,
                                              remove_last_n_in_sequence=remove_last_n_in_sequence,
                                              behavior_ids=behavior_ids, valid_only=valid_only)
        self.temporal_dilation = temporal_dilation
        self.n_frames_per_sample = n_frames_per_sample

        # Compute size of temporal window in terms of number of timesteps/frames
        window_size = temporal_dilation * (n_frames_per_sample - 1) + 1

        # Compile datapaths into list
        # sequences is a list where each element is a list of frame paths
        # Example: sequences[0][0] = 'path/to/frame'
        if behavior_ids is None:
            sequence_dir_paths = {}  # Maps sequence dir name to list of filepaths
            for frame_path in self.filenames:
                cur_dir_name = os.path.dirname(frame_path)
                cur_frame_paths = sequence_dir_paths.get(cur_dir_name, [])
                cur_frame_paths.append(frame_path)
                sequence_dir_paths[cur_dir_name] = cur_frame_paths
            self.sequences = [sorted(frame_paths) for frame_paths in sequence_dir_paths.values()]
            self.modify_path = lambda x: x
        else:
            cache_filepath = os.path.join(root_dir, 'dataset_cache.p')
            assert os.path.isfile(cache_filepath)
            print('Loading from cache (ignoring dataset parameters)')
            with open(cache_filepath, 'rb') as f:
                cache = pickle.load(f)
            self.sequences = []
            for behavior_id in self.behavior_ids:
                self.sequences.extend(cache.get(behavior_id, []))
            self.modify_path = lambda x: prepend_dataset_root(self.cfg, x)

        # Remove sequences which are too small
        valid_sequences = []
        for seq in self.sequences:
            if (len(seq) - window_size + 1) > 0:
                valid_sequences.append(seq)
        self.sequences = sorted(valid_sequences)

        self.sequence_sizes = [len(frame_paths) - window_size + 1 for frame_paths in self.sequences]

    def __len__(self):
        return sum(self.sequence_sizes)

    def _get_item(self, idx):
        for seq_idx, numel in enumerate(self.sequence_sizes):
            if idx >= numel:
                idx -= numel
            else:
                break
        filenames = [self.sequences[seq_idx][idx + i * self.temporal_dilation]
                     for i in range(self.n_frames_per_sample)]
        frame_data_list = [self.get_frame_data(self.modify_path(f)) for f in filenames]
        return frame_data_list

    def is_valid_sample(self, item):
        return True

    def __getitem__(self, idx):
        item = self._get_item(idx)

        # If the item is not a valid sample, keep sampling random items until we get a valid sample
        while not self.is_valid_sample(item):
            random_idx = random.randint(0, len(self) - 1)
            item = self._get_item(random_idx)
        return item
