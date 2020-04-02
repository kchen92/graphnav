from __future__ import print_function

import numpy as np
import os
import pickle
import torch
import torchvision.transforms as transforms

from semnav.config import get_config
from semnav.dataset.depth_transform import DepthTransform
from semnav.dataset.dataset_utils import get_frame_idx, prepend_dataset_root
from semnav.lib.categories import affordance_types
from semnav.lib.utils import load_frame
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """Base class for trajectory datasets (FrameByFrameDataset, TemporalDataset, etc.).
    """

    NO_BEHAVIOR_TOKEN = ' '
    NO_AFFORDANCES_TOKEN = ['no_affordances']
    NO_AFFORDANCES_VEC = np.asarray([-1. for idx in range(len(affordance_types))]).astype(np.float32)
    NO_NODE_NAME = ' '

    def __init__(self, root_dir, first_n_in_sequence=None, remove_last_n_in_sequence=None,
                 behavior_ids=None, valid_only=True):
        """Constructor.

        Args:
            first_n_in_sequence: This dataset will only use the first first_n_in_sequence frames of
                    each sequence. All other frames in each sequence will be discarded.
            remove_last_n_in_sequence: If there are 40 frames and remove_last_n_in_sequence == 1,
                    then only the very last frame will be removed.
            behavior_ids: List of behavior IDs which are valid.
            valid_only: Boolean denoting whether to use only frames which have been marked as valid.
        """
        self.first_n_in_sequence = first_n_in_sequence
        self.remove_last_n_in_sequence = remove_last_n_in_sequence
        if (behavior_ids is None) or isinstance(behavior_ids, list):
            self.behavior_ids = behavior_ids
        else:  # behavior_ids is a single behavior string (e.g. 'tr')
            self.behavior_ids = [behavior_ids]
        self.valid_only = valid_only

        cfg = get_config()
        self.cfg = cfg

        # Transforms
        from_numpy_transform = transforms.Lambda(lambda x: torch.from_numpy(x))
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            ])
        self.depth_transform = transforms.Compose([
            transforms.Lambda(lambda x: np.expand_dims(x, axis=0)),  # Add a channel dimension
            DepthTransform(min_depth=cfg.min_depth, max_depth=cfg.max_depth, hole_val=cfg.hole_val),
            from_numpy_transform,
            ])
        self.scan_transform = transforms.Compose([
            transforms.Lambda(lambda x: np.where(np.isnan(x), 0., x)),  # Set nans to 0.
            from_numpy_transform,
            ])

        cache_filepath = os.path.join(root_dir, 'dataset_cache.p')
        if os.path.isfile(cache_filepath):
            print('    Loading from cache (ignoring dataset parameters):', cache_filepath)
            with open(cache_filepath, 'rb') as f:
                cache = pickle.load(f)
            if self.behavior_ids is None:
                self.filenames = cache['sequential_frames']
            else:
                self.filenames = []
                for behavior_id in self.behavior_ids:
                    for seq in cache.get(behavior_id, []):
                        self.filenames.extend(seq)
                self.filenames.sort()

            self.filenames = [prepend_dataset_root(cfg, path) for path in self.filenames]
        else:
            self.sequence_to_n_frames = {}  # Includes invalid frames in the count
            self.filenames = []
            for dir_path, dir_names, filenames in os.walk(root_dir):
                for cur_filename in filenames:
                    if self.is_valid_filename(dir_path, cur_filename):
                        cur_filepath = os.path.join(dir_path, cur_filename)
                        self.filenames.append(cur_filepath)
            assert len(self.filenames) > 0
            self.filenames.sort()

    def is_first_n_in_sequence(self, fname):
        """Return whether or not the current filename is among the first n frames of the sequence.

        Args:
            fname: Filename of current file within the sequence directory.
        """
        cur_idx_in_sequence = get_frame_idx(fname)
        # Use <= instead of < because the data is 1-indexing not 0-indexing
        is_first_n = cur_idx_in_sequence <= self.first_n_in_sequence
        return is_first_n

    def is_last_n_in_sequence(self, sequence_path, fname):
        """Return whether or not the current filename is among the first n frames of the sequence.

        Args:
            sequence_path: Full path to the sequence directory.
            fname: Filename of current file within the sequence directory.
        """
        cur_idx_in_sequence = get_frame_idx(fname)

        # Remember that the frame (in sequence) idx uses 1-indexing not 0-indexing
        n_frames_in_sequence = self.sequence_to_n_frames[sequence_path]
        is_last_n = ((n_frames_in_sequence - cur_idx_in_sequence) < self.remove_last_n_in_sequence)
        return is_last_n

    def is_valid_filename(self, sequence_path, fname):
        """Return True if this is a valid filename (one that we care about). Otherwise, return
        False.

        Args:
            sequence_path: Full path to the sequence directory.
            fname: Filename of current file within the sequence directory.
        """
        if fname.endswith('.npz') is False:
            return False

        # Find number of frames in current sequence
        if sequence_path not in self.sequence_to_n_frames:
            self.sequence_to_n_frames[sequence_path] = len(os.listdir(sequence_path))

        if (self.first_n_in_sequence is not None) and not self.is_first_n_in_sequence(fname):
            return False
        if ((self.remove_last_n_in_sequence is not None)
                and self.is_last_n_in_sequence(sequence_path, fname)):
            return False
        if (self.valid_only is True) or (self.behavior_ids is not None):
            frame_path = os.path.join(sequence_path, fname)
            frame_data = self.get_frame_data(frame_path)

            if (self.valid_only is True) and (frame_data['is_invalid'] == 1):
                return False

            if ((self.behavior_ids is not None) and (frame_data['behavior_id'] not in
                                                     self.behavior_ids)):
                return False
        return True

    @classmethod
    def affordance_list2vec(cls, affordance_list):
        if affordance_list == cls.NO_AFFORDANCES_TOKEN:
            return cls.NO_AFFORDANCES_VEC

        # Convert s_l, s_r to s
        affordance_list = ['s' if (affordance == 's_l' or affordance == 's_r')
                           else affordance for affordance in affordance_list]
        new_affordance_list = []
        for affordance in affordance_list:
            if (affordance == 's_l' or affordance == 's_r'):
                cur_affordance = 's'
            elif (affordance == 'fd_l' or affordance == 'fd_r'):
                cur_affordance = 'fd'
            else:
                cur_affordance = affordance
            new_affordance_list.append(cur_affordance)

            # Print error (just in case)
            if cur_affordance not in affordance_types:
                print('Unrecognized affordance type: {}'.format(affordance))

        vec = [affordance_types[idx] in affordance_list for idx in range(len(affordance_types))]
        return np.asarray(vec).astype(np.float32)

    @staticmethod
    def affordance_vec2list(affordance_vec):
        """Convert from an affordance vector to affordance list.

        Args:
            affordance_vec: Affordance vector.
        """
        if affordance_vec.ndim == 2:
            affordance_vec = affordance_vec[0]  # Remove batch dimension
        affordance_list = []
        if affordance_vec[0] != -1:
            for idx in range(len(affordance_types)):
                if affordance_vec[idx] != 0.:
                    affordance_list.append(affordance_types[idx])
        return affordance_list

    def get_frame_data(self, frame_fpath):
        data = load_frame(frame_fpath)
        if 'localized_behavior_id' in data:
            localized_node_name, localized_behavior_id = data['localized_behavior_id'].split(' ')
        else:
            localized_node_name = self.NO_NODE_NAME
            localized_behavior_id = self.NO_BEHAVIOR_TOKEN
        processed = {
            'depth': self.depth_transform(data['depth']),
            'rgb': self.rgb_transform(data['rgb']),
            'vel': torch.from_numpy(data['vel']),
            'area_name': data.get('area_name'),
            'room_name': data['room_name'],
            'is_invalid': 1 if data.get('is_invalid', False) is True else False,
            'position': data['gt_odom']['position'],  # For debugging only
            'orientation': data['gt_odom']['orientation'],  # For junction behavior detector
            'sequence_idx': data['sequence_idx'],  # For debugging only
            'behavior_id': data.get('behavior_id', self.NO_BEHAVIOR_TOKEN),
            'localized_node_name': localized_node_name,  # This is used as a sanity check and shouldn't actually be used (use node_name) instead
            'localized_behavior_id': localized_behavior_id,
            'affordance_vec': self.affordance_list2vec(data.get('affordance_list', self.NO_AFFORDANCES_TOKEN)),
            'node_name': data.get('node_name', self.NO_NODE_NAME),
            'phase': data.get('phase', -1.),
            'frame_path': frame_fpath,  # Used for building cache
            }
        if processed['localized_behavior_id'] != self.NO_BEHAVIOR_TOKEN:
            assert processed['localized_node_name'] == processed['node_name']
        return processed
