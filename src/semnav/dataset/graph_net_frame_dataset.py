from __future__ import print_function

from torch.utils.data.dataloader import default_collate

from semnav.dataset.frame_by_frame_dataset import FrameByFrameDataset
from semnav.dataset.graph_net_dataset import GraphNetDataset, graph_net_collate_fn


class GraphNetFrameDataset(FrameByFrameDataset):
    """Dataset used for training the graph networks. This is an extension of the temporal dataset.
    """

    def __init__(self, root_dir, first_n_in_sequence=None, remove_last_n_in_sequence=None,
                 behavior_ids=None, valid_only=True):
        assert behavior_ids is None
        super(GraphNetFrameDataset, self).__init__(
            root_dir=root_dir,
            first_n_in_sequence=first_n_in_sequence,
            remove_last_n_in_sequence=remove_last_n_in_sequence,
            behavior_ids=behavior_ids,
            valid_only=valid_only
            )

        self.sem_graphs = GraphNetDataset._load_graphs(self.cfg.maps_root)

    def __getitem__(self, idx):
        """Get item. The returned subgraph information is based on the CURRENT frame and the ground
        truth node and edge is also for the CURRENT frame. This dataset should be used for
        EVALUATING graph net models and the provided subgraph information should be used for
        INITIALIZING the policy rollout. After providing graph information at the beginning of the
        episode, the future subgraphs input into the graph network/policy network should be cropped
        based on the model's predicted localization.

        Returns:
            item: Data item.
            node_idx_matrix: Matrix specifying the room category for each node.
            edge_idx_matrix: Matrix specifying the behavior category for each edge.
            edge_connections: Edge connections.
            gt_node_idx: Index of the ground truth current node in the node name list.
        """
        item = super(GraphNetFrameDataset, self).__getitem__(idx)  # item is a list of frame data dicts

        if GraphNetDataset._is_valid_sample(item, self.NO_NODE_NAME, self.NO_BEHAVIOR_TOKEN) is False:
            return item  # Cannot evaluate / use criterion on this sample

        area_graph = self.sem_graphs[item['area_name']]
        cur_gt_node_name = item['node_name']
        cur_gt_behavior_id = item['localized_behavior_id']
        (node_name_list, node_categories, edge_categories, edge_connections, node_idx,
         edge_idx) = GraphNetDataset.get_graph_data(area_graph,
                                                    subgraph_center_node_name=cur_gt_node_name,
                                                    subgraph_center_behavior_id=cur_gt_behavior_id,
                                                    n_neighbor_dist=self.cfg.n_neighbor_dist,
                                                    cur_gt_node_name=cur_gt_node_name,
                                                    cur_gt_behavior_id=cur_gt_behavior_id,
                                                    use_augmentation=self.cfg.use_gn_augmentation)
        return item, node_name_list, node_categories, edge_categories, edge_connections, node_idx, edge_idx

    @staticmethod
    def collate_fn(batch):
        assert len(batch) == 1

        if isinstance(batch[0], dict):
            return default_collate(batch)

        return graph_net_collate_fn(batch)
