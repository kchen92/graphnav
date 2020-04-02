"""Get train/val/test splits for a given dataset.
"""

from __future__ import print_function

import os
import shutil
import random

from semnav.lib.utils import mkdir
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset


def get_split_sizes(dataset_len, split_portions):
    """Return sizes of each split given the full dataset size/length and the split portions.

    Args:
        dataset_len: Size of dataset (number of samples).
        split_portions: Fraction of each size of the split. Example: (0.8, 0.1, 0.1).

    Returns:
        lengths: List with each index containing the size (number of samples) of that split.
    """
    assert sum(split_portions) == 1.
    # Determine sizes of each split
    lengths = [int(split_frac * dataset_len) for split_frac in split_portions]
    leftover = dataset_len - sum(lengths)
    lengths[0] += leftover  # Add leftover samples to the first split
    return lengths


def create_splits(root_dir, split_names=('train', 'val', 'test'), split_portions=(0.8, 0.1, 0.1),
                  seed=123):
    """Assume that the root_dir contains a list of subdirs. This function will move every subdir
    into a 'train', 'val', or 'test' folder under the root_dir, according to the split_portions.

    Args:
        split_names: Names of each split. A new subdirectory under root_dir will be created using
                each of these names. The length of split_names must match that of split_portions.
    """
    subdirs = os.listdir(root_dir)
    for cur_subdir in subdirs:
        assert os.path.isdir(os.path.join(root_dir, cur_subdir))
    for split_name in split_names:
        assert split_name not in subdirs

    dataset_len = len(subdirs)
    lengths = get_split_sizes(dataset_len, split_portions)  # Get the size of each split

    # Create a permutation
    rand = random.Random(seed)
    perm = range(dataset_len)
    rand.shuffle(perm)  # Shuffle in place

    # split_subdirs = []
    for split_idx, (offset, length) in enumerate(zip(_accumulate(lengths), lengths)):
        indices_of_subdirs_in_cur_split = perm[offset - length:offset]
        # subdirs_in_cur_split = []

        # Create split directory
        dest = os.path.join(root_dir, split_names[split_idx])  # e.g. <root_dir>/train
        print('Creating:', dest)
        mkdir(dest)

        for subdir_idx in indices_of_subdirs_in_cur_split:
            cur_subdir = subdirs[subdir_idx]
            # subdirs_in_cur_split.append(cur_subdir)

            # Move the directory
            src = os.path.join(root_dir, cur_subdir)
            shutil.move(src, dest)
        # split_subdirs.append(subdirs_in_cur_split)


def get_splits(dataset, split_portions=(0.8, 0.1, 0.1), seed=123):
    """Return a list of Subsets given a dataset. Use a fixed seed (and split portions) to ensure the
    splits are always the same across every run.
    """
    dataset_len = len(dataset)
    lengths = get_split_sizes(dataset_len, split_portions)  # Get the size of each split

    # Create a permutation
    rand = random.Random(seed)
    perm = range(dataset_len)
    rand.shuffle(perm)  # Shuffle in place

    assert sum(lengths) == dataset_len
    return [Subset(dataset, perm[offset - length:offset])
            for offset, length in zip(_accumulate(lengths), lengths)]


def test_get_splits():
    from semnav.dataset.frame_by_frame_dataset import FrameByFrameDataset
    root_dir = '/home/kchen92/Data/sem-nav/trajectory-data/nav_corridor_area_1'
    dataset = FrameByFrameDataset(root_dir)
    subsets = get_splits(dataset)


def test_create_splits():
    root_dir = '/home/kchen92/Data/sem-nav/trajectory-data/nav_corridor_area_1'
    subsets = create_splits(root_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root_dir', type=str)

    args = parser.parse_args()
    print('About to move directories in:')
    print(args.dataset_root_dir)
    subdirs = os.listdir(args.dataset_root_dir)
    print('')
    print('Some subdirectories include:')
    for idx, cur_subdir in enumerate(subdirs):
        print('\t{}'.format(cur_subdir))
        if (idx + 1) % 10 == 0:
            break
    raw_input('Are you sure you want to continue? (ctrl+C to quit)')

    create_splits(args.dataset_root_dir)
