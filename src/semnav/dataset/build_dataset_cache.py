from __future__ import print_function

import os
import pickle

from semnav.config import get_config
from semnav.dataset import load_dataset_splits
from semnav.dataset.dataset_utils import remove_dataset_root
from torch.utils.data import DataLoader


class DatasetCacheWriter(object):
    """Build a cache directory (and populate it) to quickly load different types of datasets.

    This allows us to quickly:
    - Load a sequence at a time by loading one frame at a time sequentially. Each data item is a
      frame in the dataset.
    - Load a segments of sequences for a specific behavior (e.g. turn right). Each data item is a
    sequence of chronologically adjacent frames for that behavior.

    Q:  When should I rebuild the cache?
    A:  1. The cache should be rebuilt every time you want to use the cache for loading datasets with
           different TrajectoryDataset parameters.
        2. The cache should also be rebuilt if any of the parsed bag directories move, since the
           cache relies on absolute paths.

    To delete the cache, just delete the cache pickle files from the train/val/test subdirectories.
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.cfg = get_config()

        print('Building cache:', self.root_dir)
        assert self.cfg.dataset_type == 'frame_by_frame', 'Please use a FrameByFrameDataset type'
        dataset_splits = load_dataset_splits(root_dir, self.cfg)
        split_names = ('train', 'val', 'test')
        for split_name, dataset in zip(split_names, dataset_splits):
            print('Processing {} split'.format(split_name))
            cur_cache = self.build_cache(dataset)

            # Write cache
            p_filepath = os.path.join(root_dir, split_name, 'dataset_cache.p')
            with open(p_filepath, 'wb') as f:
                pickle.dump(cur_cache, f)
            print('    Wrote to {}'.format(p_filepath))

    def build_cache(self, dataset):
        # Add dataset filenames to cache
        # Use relative paths (relative to cfg.dataset_root)
        relative_filepaths = [remove_dataset_root(self.cfg, path) for path in dataset.filenames]
        cache = {
            'sequential_frames': relative_filepaths,
            }

        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
        last_frame = None
        cur_behavior_sequence = []

        # Variables for caching for node sequences
        prev_node_name = None
        cur_frame_paths = None
        node_sequences = []

        for cur_frame in data_loader:
            cur_fpath = cur_frame['frame_path'][0]
            cur_behavior_id = cur_frame['behavior_id'][0]
            if last_frame is not None:
                last_behavior_id = last_frame['behavior_id'][0]

            if (last_frame is None) or (cur_behavior_id == last_behavior_id):
                if cur_behavior_id != dataset.NO_BEHAVIOR_TOKEN:
                    cur_behavior_sequence.append(remove_dataset_root(self.cfg, cur_fpath))
            else:
                if last_behavior_id != dataset.NO_BEHAVIOR_TOKEN:
                    if last_behavior_id not in cache:
                        cache[last_behavior_id] = []
                    cache[last_behavior_id].append(cur_behavior_sequence)
                cur_behavior_sequence = [remove_dataset_root(self.cfg, cur_fpath)]

            # Cache node sequences (similar to in phase_net_dataset)
            cur_node_name = cur_frame['node_name']
            if cur_node_name != dataset.NO_NODE_NAME:
                if cur_node_name == prev_node_name:
                    cur_frame_paths.append(remove_dataset_root(self.cfg, cur_fpath))
                else:
                    if cur_frame_paths is not None:
                        node_sequences.append(cur_frame_paths)
                    cur_frame_paths = [remove_dataset_root(self.cfg, cur_fpath)]
            prev_node_name = cur_node_name

            # Update
            last_frame = cur_frame
        cache['node_sequences'] = node_sequences
        return cache


def main():
    DatasetCacheWriter('/data/sem-nav/trajectory-data/tiny_dataset_nav_room_area_1_set_1')


if __name__ == '__main__':
    main()
