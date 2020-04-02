from __future__ import print_function
from __future__ import division

import torch

from semnav.dataset.graph_net_dataset import GraphNetDataset
from semnav.dataset.graph_net_frame_dataset import GraphNetFrameDataset
from semnav.lib.categories import BehaviorCategory
from semnav.learning import decode_batch
from semnav.learning.evaluator import Evaluator


class GraphNetEvaluator(Evaluator):

    def __init__(self, cfg=None):
        super(GraphNetEvaluator, self).__init__(cfg=cfg)
        self.sem_graphs = GraphNetDataset._load_graphs(self.cfg.maps_root)

    @staticmethod
    def get_decode_type(decode_type):
        if decode_type == 'temporal':
            return 'temporal_graph_net'
        elif decode_type == 'single_frame':  # Used by the graph net evaluator
            return 'single_frame_graph_net'
        else:
            raise ValueError

    @staticmethod
    def construct_graph_net_input(cfg, cur_area_graph, last_predicted_node_name,
                                  last_predicted_behavior_id, decode_type, batch_data):
        # Construct the subgraph
        (node_name_list, node_categories, edge_categories, edge_connections,
         node_idx, edge_idx) = GraphNetDataset.get_graph_data(
            area_graph=cur_area_graph,
            subgraph_center_node_name=last_predicted_node_name,
            subgraph_center_behavior_id=last_predicted_behavior_id,
            n_neighbor_dist=cfg.n_neighbor_dist,
            )
        tmp_batch = [[batch_data, node_name_list, node_categories,
                      edge_categories, edge_connections, node_idx, edge_idx]]
        tmp_batch = GraphNetFrameDataset.collate_fn(tmp_batch)
        tmp_vel, _ = decode_batch(tmp_batch, decode_type, cfg)
        cur_graph_net_input, _ = tmp_vel
        return cur_graph_net_input

    def predict(self, net, depth, is_new_episode):
        """The output of the GraphNet is global, node, and edge features. In addition to the
        output of the GraphNet, this method will compute and return the node name and edge behavior
        predicted by the GraphNet.
        """
        output = super(GraphNetEvaluator, self).predict(net, depth, is_new_episode)

        # Extract the predicted node name and edge name from the network output
        global_features, node_features, edge_features = output
        graph_net_input = depth['graph_net_input']
        node_names = graph_net_input['node_names']
        edge_categories = graph_net_input['edge_categories']
        edge_connections = graph_net_input['edge_connections']

        # Update the last predicted node name and last predicted behavior ID
        assert self.cfg.gn_classification == 'edge'
        assert edge_features.size(1) == 1
        predicted_edge_idx = int(torch.squeeze(torch.argmax(edge_features, dim=0)))
        behavior_category_enum = BehaviorCategory(int(edge_categories[predicted_edge_idx]))
        last_predicted_behavior_id = behavior_category_enum.name
        predicted_node_idx = edge_connections[predicted_edge_idx, 0]
        last_predicted_node_name = node_names[predicted_node_idx]

        output_dict = {
            'global_features': global_features,
            'node_features': node_features,
            'edge_features': edge_features,
            'node_names': node_names,
            'edge_categories': edge_categories,
            'edge_connections': edge_connections,
            }

        return output_dict, last_predicted_node_name, last_predicted_behavior_id

    def prediction2str(self, net_output, last_predicted_node_name, last_predicted_behavior_id):
        pred_str = last_predicted_node_name + ' ' + last_predicted_behavior_id
        return net_output, pred_str


if __name__ == '__main__':
    GraphNetEvaluator().test()
