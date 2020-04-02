from __future__ import print_function

import argparse
import os
import torch

from pprint import pprint

from semnav.lib.utils import mkdir


DATASET_ROOT = '/data/sem-nav/trajectory-data'
LOG_ROOT = '/data/sem-nav/experiments/public'
STANFORD_JSON = '/home/kchen92/Dev/sem-nav/data/semantic_labels.json'
MAPS_ROOT = '/home/kchen92/Dev/sem-nav/maps/v0.2'


parser = argparse.ArgumentParser(description='Semantic Navigation Configuration')
parser.add_argument('--dataset', type=str)
parser.add_argument('--log_dir', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--ckpt_path', type=str)  # Load ckpt from this path  # Unused for now
parser.add_argument('--print_freq', default=500, type=int)
parser.add_argument('--ckpt_freq', type=int)
parser.add_argument('--val_freq', default=500, type=int)
parser.add_argument('--n_epochs', default=20, type=int)
parser.add_argument('--n_workers', default=4, type=int)

# Depth sensor
parser.add_argument('--min_depth', default=0.8, type=float)
parser.add_argument('--max_depth', default=3.5, type=float)
parser.add_argument('--hole_val', default=0., type=float)

# Dataset parameters
parser.add_argument('--dataset_type', type=str)
parser.add_argument('--first_n_frames', type=int)
parser.add_argument('--remove_last_n_frames', type=int)
parser.add_argument('--ignore_validity', action='store_true')
parser.add_argument('--behavior_id', type=str)  # Not affected by cache
parser.add_argument('--temporal_dilation', default=1, type=int)  # Temporal dataset only
parser.add_argument('--n_frames_per_sample', default=20, type=int)  # Temporal dataset only

# Learning
parser.add_argument('--behaviornet_type', type=str)  # Only used for train/test
parser.add_argument('--use_semantic_class', type=str)  # Adds semantics to input channel

# Graph/GraphNet
parser.add_argument('--n_neighbor_dist', type=int, default=2)  # For subgraph construction
parser.add_argument('--aggregate_method', type=str, default='sum', choices=['sum', 'avg'])
parser.add_argument('--gn_classification', type=str, default='edge', choices=['node', 'edge', 'joint'])
parser.add_argument('--use_gn_augmentation', action='store_true')
parser.add_argument('--n_graphnet_layers', type=int, default=2)
parser.add_argument('--graphnet_feat_size', type=int, default=256)

# GraphNet using affordances
parser.add_argument('--gn_cnn_encoder_learning_rate', type=float)

# Only used when running the evaluator standalone, otherwise ignored
parser.add_argument('--visualize_results', action='store_true')

args = parser.parse_args()
args.dataset_root = DATASET_ROOT
args.stanford_json = STANFORD_JSON
args.maps_root = MAPS_ROOT

print('---------------- CONFIG ----------------')
pprint(vars(args))
print('----------------------------------------')


def get_config():
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args.valid_only = not args.ignore_validity

    # Set oor to be equal to tr when segmenting behaviors
    if args.behavior_id == 'tr':
        args.behavior_id = ['tr', 'rio']
    elif args.behavior_id == 'tl':
        args.behavior_id = ['tl', 'lio']

    if args.log_dir is not None:
        args.log_dir = os.path.join(LOG_ROOT, args.log_dir)
        args.ckpt_dir = os.path.join(args.log_dir, 'models')  # Directory for saving checkpoints

        # Make directories
        mkdir(args.log_dir)
        mkdir(args.ckpt_dir)

    return args


if __name__ == '__main__':
    get_config()
