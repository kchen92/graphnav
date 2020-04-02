from __future__ import print_function

import os

from semnav.config import get_config
from semnav.dataset.frame_by_frame_dataset import FrameByFrameDataset
from semnav.dataset.temporal_dataset import TemporalDataset
from semnav.dataset.graph_net_dataset import GraphNetDataset
from semnav.dataset.graph_net_frame_dataset import GraphNetFrameDataset
from torch.utils.data import ConcatDataset


def load_dataset(data_dir, cfg):
    """Load a dataset given a parsed rosbag directory.

    Args:
        data_dir: Full path to the data directory.
        cfg: Configuration.
    """
    print('    Loading:', data_dir)
    if cfg.dataset_type == 'frame_by_frame':
        return FrameByFrameDataset(data_dir, first_n_in_sequence=cfg.first_n_frames,
                                   remove_last_n_in_sequence=cfg.remove_last_n_frames,
                                   behavior_ids=cfg.behavior_id, valid_only=cfg.valid_only)
    elif cfg.dataset_type == 'temporal':
        return TemporalDataset(data_dir, temporal_dilation=cfg.temporal_dilation,
                               n_frames_per_sample=cfg.n_frames_per_sample,
                               first_n_in_sequence=cfg.first_n_frames,
                               remove_last_n_in_sequence=cfg.remove_last_n_frames,
                               behavior_ids=cfg.behavior_id, valid_only=cfg.valid_only)
    elif cfg.dataset_type == 'graph_net':
        return GraphNetDataset(data_dir, temporal_dilation=cfg.temporal_dilation,
                               n_frames_per_sample=cfg.n_frames_per_sample,
                               first_n_in_sequence=cfg.first_n_frames,
                               remove_last_n_in_sequence=cfg.remove_last_n_frames,
                               behavior_ids=cfg.behavior_id, valid_only=cfg.valid_only)
    elif cfg.dataset_type == 'single_frame_graph_net':
        return GraphNetFrameDataset(data_dir, first_n_in_sequence=cfg.first_n_frames,
                                    remove_last_n_in_sequence=cfg.remove_last_n_frames,
                                    behavior_ids=cfg.behavior_id, valid_only=cfg.valid_only)
    else:
        raise ValueError('Please select a valid dataset type')


def load_dataset_splits(bag_dir_name, cfg):
    """Load the train/val/test splits as Datasets for a parsed bag directory.

    Args:
        bag_dir_name: Name of bag/directory under cfg.dataset_root.
        cfg: Configuration.
    """
    processed_bag_dir = os.path.join(cfg.dataset_root, bag_dir_name)
    train_set = load_dataset(os.path.join(processed_bag_dir, 'train'), cfg)
    val_set = load_dataset(os.path.join(processed_bag_dir, 'val'), cfg)
    test_set = load_dataset(os.path.join(processed_bag_dir, 'test'), cfg)
    return train_set, val_set, test_set


def concat_datasets(bag_names, cfg):
    """Combine the train, val, and test sets from each parsed bag directory.

    Args:
        bag_names: List of bag/directory names under cfg.dataset_root.
        cfg: Configuration.
    """
    train_sets, val_sets, test_sets = ([], [], [])
    for bag_name in bag_names:
        cur_train_set, cur_val_set, cur_test_set = load_dataset_splits(bag_name, cfg)
        train_sets.append(cur_train_set)
        val_sets.append(cur_val_set)
        test_sets.append(cur_test_set)
    train_set = ConcatDataset(train_sets)
    val_set = ConcatDataset(val_sets)
    test_set = ConcatDataset(test_sets)
    return train_set, val_set, test_set


def merge_into_single_split(bag_names, cfg):
    """Each parsed bag directory has train/val/test subfolders. This function can be used to combine
    the data across all three splits (train/val/test) into a single torch dataset/split.
    """
    train_set, val_set, test_set = concat_datasets(bag_names, cfg)
    dataset = ConcatDataset([train_set, val_set, test_set])
    return dataset


def get_split_datasets(dataset_name):
    """Returns a train set, val set, and test set.
    """
    print('loading dataset...')
    cfg = get_config()
    if dataset_name == 'v0.2':
        train_bag_names = [
            'v0.2/nav_area_1',
            'v0.2/nav_area_5b',
            'v0.2/nav_area_6',
            ]
        val_bag_names = [
            'v0.2/nav_area_3',
            ]
        test_bag_names = [
            'v0.2/nav_area_4',
            ]
        train_set = merge_into_single_split(train_bag_names, cfg)
        val_set = merge_into_single_split(val_bag_names, cfg)
        test_set = merge_into_single_split(test_bag_names, cfg)
    else:
        print('Selected dataset: %s' % dataset_name)
        raise ValueError('Please use a valid dataset.')

    print('full dataset size:', len(train_set) + len(val_set) + len(test_set))
    print('train set size:', len(train_set))
    print('val set size:', len(val_set))
    print('test set size:', len(test_set))
    print('')
    return train_set, val_set, test_set
