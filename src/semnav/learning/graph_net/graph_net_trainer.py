from __future__ import print_function
from __future__ import division

import torch

from semnav.learning.trainer import Trainer
from semnav.learning.graph_net.graph_net_evaluator import GraphNetEvaluator


class GraphNetTrainer(Trainer):

    def __init__(self):
        super(GraphNetTrainer, self).__init__()

    @staticmethod
    def get_decode_type(decode_type):
        return GraphNetEvaluator.get_decode_type(decode_type)

    @staticmethod
    def edge_classification_criterion(outputs, target_output_dict):
        """Computes the training loss for the GraphNet.

        Args:
            outputs: Outputs from the graph network.
            target_output_dict: Dict containing information such as the ground truth node index and
                    number of nodes per graph.
        """
        # Parse inputs
        global_features, node_features, edge_features = outputs
        assert edge_features.size(1) == 1

        device = edge_features.device

        graph_net_input, target_output = target_output_dict
        gt_edge_idx_list = target_output['gt_edge_idx']
        n_edges_per_graph_list = target_output['n_edges_per_graph']

        # Construct tensors for computing cross entropy loss
        # The prediction matrix should have shape (N, C) where N is the batch size and C is the
        # number of classes
        batch_size = len(n_edges_per_graph_list)
        n_cols = max(n_edges_per_graph_list)
        prediction_matrix = torch.zeros(batch_size, n_cols, device=device)
        start_idx = 0
        target_list = []
        for idx_in_batch, (gt_edge_idx, n_edges_in_graph) in enumerate(zip(gt_edge_idx_list,
                                                                           n_edges_per_graph_list)):
            # Ground truth node index
            gt_edge_idx_for_sample = gt_edge_idx - start_idx
            assert (gt_edge_idx_for_sample >= 0) and (gt_edge_idx_for_sample < n_edges_in_graph)
            target_list.append(gt_edge_idx_for_sample)

            # Set up prediction matrix
            end_idx = start_idx + n_edges_in_graph
            prediction_matrix[idx_in_batch, :n_edges_in_graph] = edge_features[start_idx:end_idx, 0]

            # Update start_idx for the next iteration
            start_idx += n_edges_in_graph
        target = torch.tensor(target_list, dtype=torch.int64, device=device)

        # Compute cross entropy loss
        losses = torch.empty(batch_size, dtype=torch.float32, device=device)
        for idx_in_batch in range(batch_size):
            cur_loss = (-prediction_matrix[idx_in_batch, target[idx_in_batch]]
                        + torch.log(torch.sum(torch.exp(prediction_matrix[idx_in_batch, :(n_edges_per_graph_list[idx_in_batch])]))))
            losses[idx_in_batch] = cur_loss
        total_loss = torch.mean(losses)
        return total_loss

    def get_criterion(self):
        assert self.cfg.gn_classification == 'edge'
        return self.edge_classification_criterion

    def get_evaluator(self):
        return GraphNetEvaluator()


if __name__ == '__main__':
    GraphNetTrainer().train()
