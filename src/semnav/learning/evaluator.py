from __future__ import print_function
from __future__ import division

import os
import torch

from semnav.config import get_config
from semnav.dataset import get_split_datasets
from semnav.dataset.dataset_utils import get_frame_idx
from semnav.dataset.dataset_visualizer import DatasetVisualizer
from semnav.dataset.graph_net_frame_dataset import GraphNetFrameDataset
from semnav.lib.categories import BehaviorCategory
from semnav.learning.behavior_net.behavior_cnn import BehaviorCNN
from semnav.learning.behavior_net.behavior_rnn import BehaviorRNN
from semnav.learning.graph_net.graph_net import GraphNet
from semnav.learning.phase_net.phase_rnn import PhaseRNN
from semnav.learning import decode_batch, get_net
from torch.utils.data import DataLoader


class Evaluator(object):

    def __init__(self, cfg=None):
        if cfg is None:
            self.cfg = get_config()
        else:
            self.cfg = cfg
        self.hidden = None  # Only used for BehaviorRNN
        self.depth_stack = None  # Only used for BehaviorCNN with temporal dataset (StackCNN)

        if self.cfg.visualize_results is True:
            self.dataset_visualizer = DatasetVisualizer()

    @staticmethod
    def load_eval_datasets(cfg):
        """Load datasets for evaluation. Always use FrameByFrame dataset for evaluation. This way, we
        can evaluate the different behavior networks (BehaviorCNN, BehaviorRNN, etc.) on the same
        dataset/metrics.

        Args:
            cfg: Configuration.

        Returns:
            dataset_splits: Train set, validation set, and test set.
        """
        # Temporarily change dataset type to be frame_by_frame
        cur_dataset_type = cfg.dataset_type
        if cfg.dataset_type == 'graph_net':
            cfg.dataset_type = 'single_frame_graph_net'
        else:
            cfg.dataset_type = 'frame_by_frame'

        # Get the evaluation (frame by frame) datasets
        train_set, val_set, test_set = get_split_datasets(cfg.dataset)

        # Restore dataset type
        cfg.dataset_type = cur_dataset_type
        return train_set, val_set, test_set

    @staticmethod
    def get_evaluation_batch_size():
        """Return the batch size used for evaluation. The self.evaluate_dataset() method expects a
        DataLoader with this batch size.

        Returns:
            batch_size: Batch size.
        """
        return 1

    def predict(self, net, depth, is_new_episode):
        """Make a prediction online. This will update the hidden state tracked by the
        BehaviorEvaluator if applicable.

        Args:
            net: Network.
            depth: Depth input to the network. This can be a depth stack. It should have shape
                    (batch_size x n_channels x height x width).

        Returns:
            output: Output of the network, which should be (batch_size x 2).
        """
        net.eval()  # Set to eval mode

        if is_new_episode is True:
            print('New episode!')

        # Parse depth input (since it is different when we use a GraphNet)
        if isinstance(net, GraphNet) and isinstance(depth, dict):
            depth, graph_net_input = (depth['depth'], depth['graph_net_input'])

        if ((isinstance(net, BehaviorCNN) and (self.cfg.dataset_type == 'temporal'))
                or (isinstance(net, GraphNet) and (self.cfg.dataset_type == 'graph_net'))):
            # If this is a new episode, create a repeat of the first input image
            if self.cfg.use_semantic_class is not None:
                n_channels_per_frame = 2
            else:
                n_channels_per_frame = 1
            if is_new_episode:
                self.depth_stack = depth.repeat(1, self.cfg.n_frames_per_sample, 1, 1)
            else:  # Otherwise, update the current stack of input images
                # Most recent image is at the end
                self.depth_stack = torch.cat([self.depth_stack[:, n_channels_per_frame:, :, :], depth], dim=1)
            depth = self.depth_stack

        # If this is a new episode, reset hidden state
        if (isinstance(net, BehaviorRNN) or isinstance(net, PhaseRNN)) and is_new_episode:
            print('Resetting hidden state')
            batch_size = depth.size(0)
            hidden = net.initial_hidden(batch_size=batch_size)
            # Send to device
            if isinstance(hidden, list):
                self.hidden = [x.to(self.cfg.device) for x in hidden]
            else:
                self.hidden = hidden.to(self.cfg.device)

        if isinstance(net, BehaviorCNN):
            output = net(depth)
        elif isinstance(net, GraphNet):
            output = net(depth, graph_net_input)
        elif isinstance(net, BehaviorRNN) or isinstance(net, PhaseRNN):
            output, self.hidden = net(depth, self.hidden)
        else:
            raise ValueError('Validation on this network is not supported.')
        return output

    @staticmethod
    def get_seq_path(frame_path):
        """Return the sequence name as a String for a given (full) frame path.
        """
        return os.path.dirname(frame_path)

    @classmethod
    def is_new_episode(cls, last_batch, cur_batch):
        """Check if the current batch is immediately after the last batch. This returns True if the
        current batch is in a different sequence/episode than the previous batch. This also returns
        True if there is a time gap between the last batch and the current batch.

        Assumes batches come from a FrameByFrameDataset. Also assumes a batch size of 1.
        """
        if last_batch is None:
            return True

        last_frame_path = last_batch['frame_path'][0]
        cur_frame_path = cur_batch['frame_path'][0]

        # Check if the batches are in the same episode/sequence
        last_seq_path = cls.get_seq_path(last_frame_path)
        cur_seq_path = cls.get_seq_path(cur_frame_path)
        if last_seq_path != cur_seq_path:
            return True

        # Check if the batches are adjacent in the same episode/sequence
        last_frame_idx = get_frame_idx(last_frame_path)
        cur_frame_idx = get_frame_idx(cur_frame_path)
        return last_frame_idx != (cur_frame_idx - 1)

    def evaluate_dataset(self, net, criterion, data_loader):
        """Run evaluation online by using a FrameByFrameDataset.

        Always use a decode_type of 'single_frame'. This way, we can compare different models under
        the same circumstances.

        Assumes a batch size of 1.
        """
        is_training = net.training  # Remember if the network is training or not

        net.eval()  # Set to eval mode

        running_loss = 0.
        counter = 0
        print('Make sure this is set correctly')
        decode_type = 'single_frame'  # Always use a decode_type of 'single_frame'
        use_single_frame = decode_type == 'single_frame'
        decode_type = self.get_decode_type(decode_type)
        last_batch = None
        localized = False  # For graph net eval
        last_predicted_node_name = None  # For graph net eval
        last_predicted_behavior_id = None  # For graph net eval
        if isinstance(net, PhaseRNN):
            output = torch.tensor([[0.]], dtype=torch.float32)
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                raw_batch = batch  # Only used for GraphNet
                vel, depth = decode_batch(batch, decode_type, self.cfg)

                # Sanity check
                if not isinstance(net, GraphNet):
                    # Check that batch size is 1
                    batch_size = vel.size(0)  # Compute batch size
                    assert batch_size == 1

                # Check if this is the beginning of a new episode
                if isinstance(net, PhaseRNN):
                    is_new_episode = (Evaluator.is_new_episode(last_batch, batch)
                                      or self.is_new_episode(output))
                elif isinstance(net, GraphNet):
                    if not isinstance(batch, dict):  # Graph info provided
                        has_graph_info = True
                        batch = vel[1]['batch']
                    else:
                        has_graph_info = False
                    # While we have not localized the agent in this episode yet, keep the
                    # is_new_episode flag to True. Note that localized could be set to False from a
                    # previous iteration.
                    is_new_episode = self.is_new_episode(last_batch, batch) or not localized
                else:
                    is_new_episode = self.is_new_episode(last_batch, batch)

                if is_new_episode:
                    localized = False  # For graph net eval
                    last_predicted_node_name = None  # For graph net eval
                    last_predicted_behavior_id = None  # For graph net eval

                # Update
                last_batch = batch

                # If GraphNet, check if localized (AKA if graph info has been provided)
                if isinstance(net, GraphNet) and (has_graph_info is True):
                    localized = True  # For graph net eval

                if isinstance(net, GraphNet) and (localized is True):
                    # Agent has been localized
                    # Set up the input to the network (combine it all into the depth variable)
                    if (is_new_episode is True) or True and (vel is not None):
                        cur_area_graph = self.sem_graphs[batch['area_name'][0]]
                        initial_graph_net_input, _ = vel
                        depth = {
                            'depth': depth,
                            'graph_net_input': initial_graph_net_input,
                            }
                    else:
                        # We are in the middle of a rollout for this episode. Provide the subgraph
                        # based on the previous localization prediction.

                        cur_graph_net_input = self.construct_graph_net_input(
                            self.cfg,
                            cur_area_graph,
                            last_predicted_node_name,
                            last_predicted_behavior_id,
                            decode_type,
                            batch,
                            )

                        depth = {
                            'depth': depth,
                            'graph_net_input': cur_graph_net_input,
                            }

                # Start evaluating the sequence/episode once we have found the starting node/position
                if isinstance(net, GraphNet) and (localized is False):
                    continue

                # Sanity check
                if isinstance(net, GraphNet) and (localized is False):
                    assert torch.unique(depth['graph_net_input']['graph_idx_of_node']) == 0
                    assert torch.unique(depth['graph_net_input']['graph_idx_of_edge']) == 0

                output = self.predict(net, depth, is_new_episode)
                if isinstance(net, GraphNet):
                    # Decode the output of GraphNetEvaluator.predict() and update the predicted location
                    output, last_predicted_node_name, last_predicted_behavior_id = output

                    if has_graph_info is False:
                        # We cannot compare with GT since no ground truth is provided
                        continue

                    # Use a different criterion (from training) for evaluating the GraphNet
                    # Ignore the input criterion and measure accuracy instead
                    gt_graph_net_dict, target_output = vel
                    node_names_of_gt_subgraph = gt_graph_net_dict['node_names']
                    edge_categories_of_gt_subgraph = gt_graph_net_dict['edge_categories']
                    assert len(target_output['gt_node_idx']) == 1
                    assert len(target_output['gt_edge_idx']) == 1
                    gt_node_name = node_names_of_gt_subgraph[target_output['gt_node_idx'][0]]
                    gt_behavior_category_enum = BehaviorCategory(int(edge_categories_of_gt_subgraph[target_output['gt_edge_idx'][0]]))
                    gt_behavior_id = gt_behavior_category_enum.name

                    assert self.cfg.gn_classification == 'edge'
                    loss = (gt_node_name == last_predicted_node_name) and (gt_behavior_id == last_predicted_behavior_id)
                    loss = float(loss)
                else:
                    loss = criterion(output, vel)

                # Update counters
                counter += 1
                if isinstance(loss, float):
                    running_loss += loss
                else:
                    running_loss += loss.item()

                if (i + 1) % self.cfg.print_freq == 0:
                    print('    evaluated %d iterations: %f' % (i + 1, running_loss / counter))

                # Display the prediction
                if self.cfg.visualize_results is True:
                    # Modify dataset_item to include the prediction
                    if isinstance(net, GraphNet):
                        prediction_as_str = self.prediction2str(output, last_predicted_node_name, last_predicted_behavior_id)
                    else:
                        prediction_as_str = self.prediction2str(output)

                    if isinstance(net, GraphNet) and (has_graph_info is True):  # Make sure to visualize graph info if provided
                        dataset_item = raw_batch
                        dataset_item[0]['prediction_str'] = prediction_as_str
                    else:
                        dataset_item = batch
                        dataset_item['prediction_str'] = prediction_as_str

                    # Visualize
                    to_break = self.dataset_visualizer.visualize_data_loader_item(dataset_item, use_frame_by_frame=use_single_frame)
                    if to_break:
                        break

        if is_training:
            net.train()  # Set to train mode
        return running_loss / counter
