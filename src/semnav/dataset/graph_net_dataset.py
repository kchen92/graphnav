from __future__ import print_function

import os
import torch

from random import choice
from torch.utils.data.dataloader import default_collate

from semnav.dataset.temporal_dataset import TemporalDataset
from semnav.lib.sem_graph import SemGraph
from semnav.lib.sub_graph import SubGraph


def is_same_behavior_id_str(behavior_id_str_1, behavior_id_str_2):
    """Checks if the two behavior ID strings are the same (e.g. 'fd' vs. 'fd_r' should return
    True).

    Args:
        behavior_id_str_1: Behavior ID as a string.
        behavior_id_str_2: Behavior ID as a string.
    """
    return behavior_id_str_1.startswith(behavior_id_str_2) or behavior_id_str_2.startswith(behavior_id_str_1)


class GraphNetDataset(TemporalDataset):
    """Dataset used for training the graph networks. This is an extension of the temporal dataset.

    The proper way to do this is to create a separate GraphNetDataset class which does not have any
    parent classes. Then, the temporal graph net dataset should subclass both the TemporalDataset
    and the new GraphNetDataset. Please address this later.
    """

    def __init__(self, root_dir, temporal_dilation=1, n_frames_per_sample=5,
                 first_n_in_sequence=None, remove_last_n_in_sequence=None, behavior_ids=None,
                 valid_only=True):
        assert behavior_ids is None
        super(GraphNetDataset, self).__init__(
            root_dir=root_dir,
            temporal_dilation=temporal_dilation,
            n_frames_per_sample=n_frames_per_sample,
            first_n_in_sequence=first_n_in_sequence,
            remove_last_n_in_sequence=remove_last_n_in_sequence,
            behavior_ids=behavior_ids,
            valid_only=valid_only
            )

        self.sem_graphs = GraphNetDataset._load_graphs(self.cfg.maps_root)

    @staticmethod
    def _load_graphs(maps_root):
        """Returns a dictionary of SemGraph objects using the area name as the key.
        """
        # Load the graphs for each area
        print('Loading graphs')
        area_names = ['area_1', 'area_3', 'area_4', 'area_5b', 'area_6']
        sem_graphs = {}
        for area_name in area_names:
            yaml_filepath = os.path.join(maps_root, area_name + '.yaml')
            print('--> Loading graphs using yaml file: {}'.format(yaml_filepath))
            sem_graphs[area_name] = SemGraph(yaml_filepath)
        print('--> Done loading graphs')
        return sem_graphs

    @staticmethod
    def _is_valid_sample(single_frame_item, no_name_token, no_behavior_token):
        if ((single_frame_item['node_name'] != no_name_token)
                and (single_frame_item['localized_behavior_id'] != no_behavior_token)):
            return True
        else:
            return False

    def is_valid_sample(self, item):
        """Filter out samples which do not have a tagged node name.
        """
        is_valid = super(GraphNetDataset, self).is_valid_sample(item)
        if is_valid and GraphNetDataset._is_valid_sample(item[-1], self.NO_NODE_NAME, self.NO_BEHAVIOR_TOKEN):
            return True
        else:
            return False

    @staticmethod
    def get_edge_connections(nxG, node_name_list):
        """Get a list of edge connections (edge i corresponds to connection from node A to node B).

        Args:
            nxG: NetworkX graph.

        Returns:
            connections_list: List of (x, y) tuples where x and y are the start and end node indices.
        """
        for node_name, x in zip(node_name_list, nxG.nodes):
            assert x == node_name
        node_name2node_idx = {node_name: idx for idx, node_name in enumerate(nxG.nodes)}
        connections_list = []
        for u, v in nxG.edges:
            connections_list.append((node_name2node_idx[u], node_name2node_idx[v]))
        return connections_list

    @staticmethod
    def get_graph_data(area_graph, subgraph_center_node_name, subgraph_center_behavior_id,
                       n_neighbor_dist, cur_gt_node_name=None, cur_gt_behavior_id=None,
                       use_augmentation=False):
        """Return data about the graph.

        Args:
            area_graph: SemGraph for the current area.
            subgraph_center_node_name: Name of the last (GT or predicted) node/position in the graph, used for
                    subgraph creation.
            cur_node_name: Name of the current GT node/position in the graph, used for determining
                    the GT prediction.
            n_neighbor_dist: Neighbor distance for subgraph creation.
            cur_gt_node_name: Node name of the ground truth node position. If None, the node_idx
                    output will be None.
            cur_gt_behavior_id: Behavior ID of the ground truth edge coming out of cur_gt_node_name.
                    If None, the edge_idx output will be None.

        Returns:
            node_name_list: List of node names in the subgraph.
            node_categories: Categories for each node in the subgraph.
            edge_categories: Categories for each edge in the subgraph.
            node_idx: Index of the ground truth node (corresponding with cur_gt_node_name).
            edge_idx: Index of the ground truth edge (corresponding with cur_gt_node_name and
                    cur_gt_behavior_id).
        """
        subgraph = SubGraph(area_graph, subgraph_center_node_name, n_neighbor_dist=n_neighbor_dist,
                            cur_behavior_id=subgraph_center_behavior_id)
        if use_augmentation is True:
            assert cur_gt_node_name is not None
            while True:
                if subgraph_center_behavior_id is not None:
                    new_subgraph_center_edge = choice(list(subgraph.nxG_compact.edges))
                    new_subgraph_center_behavior_id_enum = subgraph.nxG_compact.edges[new_subgraph_center_edge]['behavior_category']
                    new_subgraph_center_behavior_id = new_subgraph_center_behavior_id_enum.name
                    new_subgraph_center_node_name = new_subgraph_center_edge[0]
                else:
                    raise NotImplementedError('I havent tested this.')
                    # Choose a random node from the subgraph as the new subgraph center
                    new_subgraph_center_node_name = choice(list(subgraph.nxG_compact.nodes))
                    new_subgraph_center_behavior_id = None
                proposed_subgraph = SubGraph(area_graph, new_subgraph_center_node_name,
                                             n_neighbor_dist=n_neighbor_dist,
                                             cur_behavior_id=new_subgraph_center_behavior_id)

                # Make sure the gt node and edge are in the new subgraph
                exit_loop = False
                for cur_edge in proposed_subgraph.nxG_compact.edges:
                    if ((cur_edge[0] == cur_gt_node_name)
                            and ((cur_gt_behavior_id is None) or is_same_behavior_id_str(proposed_subgraph.nxG_compact.edges[cur_edge]['behavior_category'].name, cur_gt_behavior_id))):
                        exit_loop = True
                        break
                if exit_loop is True:
                    subgraph = proposed_subgraph
                    break

        # We need to make sure the node order and edge order is always the same when we run nxG.nodes or nxG.edges
        node_name_list = []
        node_category_list = []
        for node_name in subgraph.nxG_compact.nodes:
            node_name_list.append(node_name)
            node_category_list.append(subgraph.nxG_compact.nodes[node_name]['room_category'])
        edge_tuple_list = []
        edge_category_list = []
        node_idx = None
        edge_idx = None
        for idx, edge_tuple in enumerate(subgraph.nxG_compact.edges):
            edge_tuple_list.append(edge_tuple)
            edge_category_list.append(subgraph.nxG_compact.edges[edge_tuple]['behavior_category'])

            if ((cur_gt_node_name is not None) and (cur_gt_behavior_id is not None)
                    and (edge_tuple[0] == cur_gt_node_name)
                    and is_same_behavior_id_str(subgraph.nxG_compact.edges[edge_tuple]['behavior_category'].name, cur_gt_behavior_id)):
                edge_idx = idx
        edge_connections = GraphNetDataset.get_edge_connections(subgraph.nxG_compact, node_name_list)
        assert len(node_name_list) == len(set(node_name_list))  # Check uniqueness in order for list.index() to work properly
        if cur_gt_node_name is not None:
            node_idx = node_name_list.index(cur_gt_node_name)
        return (
            node_name_list,
            torch.tensor(node_category_list, dtype=torch.int64),
            torch.tensor(edge_category_list, dtype=torch.int64),
            torch.tensor(edge_connections, dtype=torch.int64),
            node_idx,
            edge_idx,
            )

    def __getitem__(self, idx):
        """Get item. The returned subgraph information is based on the PREVIOUS frame and the ground
        truth node and edge is for the CURRENT frame. This is because we want to predict where the
        agent is located NOW given the previous frame's localization prediction and the current
        visual input. Hmm no this doesn't seem right. We just want to predict the current position
        based on the most recent visual information.

        Returns:
            item: Data item.
            node_idx_matrix: Matrix specifying the room category for each node.
            edge_idx_matrix: Matrix specifying the behavior category for each edge.
            edge_connections: Edge connections.
            gt_node_idx: Index of the ground truth current node in the node name list.
        """
        item = super(GraphNetDataset, self).__getitem__(idx)  # item is a list of frame data dicts

        most_recent_item = item[-1]  # Use the most recent frame data: item[-1]
        area_graph = self.sem_graphs[most_recent_item['area_name']]
        cur_gt_node_name = most_recent_item['node_name']
        cur_gt_behavior_id = most_recent_item['localized_behavior_id']

        prev_gt_node_name = item[-1]['node_name']
        prev_gt_behavior_id = item[-1]['localized_behavior_id']

        (node_name_list, node_categories, edge_categories, edge_connections, node_idx,
         edge_idx) = GraphNetDataset.get_graph_data(area_graph,
                                                    subgraph_center_node_name=prev_gt_node_name,
                                                    subgraph_center_behavior_id=prev_gt_behavior_id,
                                                    n_neighbor_dist=self.cfg.n_neighbor_dist,
                                                    cur_gt_node_name=cur_gt_node_name,
                                                    cur_gt_behavior_id=cur_gt_behavior_id,
                                                    use_augmentation=self.cfg.use_gn_augmentation)
        return item, node_name_list, node_categories, edge_categories, edge_connections, node_idx, edge_idx


def graph_net_collate_fn(batch):
    """Collate function for the GraphNetDataset.

    - B: Batch size.
    - N: Total number of nodes in batch.
    - E: Total number of edges in batch.
    - D: Dimension of global, node, and edge features (assume they are the same).

    Args:
        batch: List of dataset items. For the GraphNetDataset, each dataset item is a list of length
                n_frames_per_sample.

    Returns:
        item_batch: The standard output from a TemporalDataset.
        node_names: A list of length N containing the names of each node.
        node_categories: LongTensor of shape (N,) indicating the RoomCategory IntEnum of each
                node.
        edge_categories: LongTensor of shape (E,) indicating the BehaviorCategory IntEnum of each
                edge.
        edge_connections: LongTensor of shape (E, 2) containing edge connections (a mapping from
                edges to node indices). Each element is a node index (from 0 to N-1).
        graph_idx_of_node: LongTensor of shape (N,) where each element indicates which sample
                in the batch the current node belongs to. Elements should range from 0 to B-1.
        graph_idx_of_edge: LongTensor of shape (E,) where each element indicates which sample in the
                batch the current edge belongs to. Elements should range from 0 to B-1.
    """
    standard_batch = []
    node_name_list = []  # List of all node names in the batch
    node_categories = []
    edge_categories = []
    graph_idx_of_node = []
    graph_idx_of_edge = []
    batch_edge_connections_list = []
    n_nodes_per_graph = []
    n_edges_per_graph = []
    gt_node_idx_list = []
    gt_edge_idx_list = []
    n_nodes_so_far = 0
    for graph_idx, sample in enumerate(batch):
        # Parse sample
        (item, cur_node_name_list, cur_node_categories, cur_edge_categories, edge_connections,
         gt_node_idx, gt_edge_idx) = sample

        standard_batch.append(item)
        node_categories.append(cur_node_categories)
        edge_categories.append(cur_edge_categories)
        node_name_list.extend(cur_node_name_list)
        graph_idx_of_node.append(torch.tensor([graph_idx], dtype=torch.int64).repeat(cur_node_categories.size(0)))
        graph_idx_of_edge.append(torch.tensor([graph_idx], dtype=torch.int64).repeat(cur_edge_categories.size(0)))

        # Edge connections
        n_nodes = cur_node_categories.size(0)
        n_edges = cur_edge_categories.size(0)
        batch_edge_connections = edge_connections + n_nodes_so_far
        batch_edge_connections_list.append(batch_edge_connections)
        if gt_node_idx is not None:
            gt_node_idx_list.append(gt_node_idx + n_nodes_so_far)
        if gt_edge_idx is not None:
            gt_edge_idx_list.append(gt_edge_idx + sum(n_edges_per_graph))
        n_nodes_so_far += n_nodes
        n_nodes_per_graph.append(n_nodes)
        n_edges_per_graph.append(n_edges)
    item_batch = default_collate(standard_batch)
    node_categories_tensor = torch.cat(node_categories, dim=0)
    edge_categories_tensor = torch.cat(edge_categories, dim=0)
    edge_connections_tensor = torch.cat(batch_edge_connections_list, dim=0)
    graph_idx_of_node_tensor = torch.cat(graph_idx_of_node, dim=0)
    graph_idx_of_edge_tensor = torch.cat(graph_idx_of_edge, dim=0)
    assert n_nodes_so_far == node_categories_tensor.size(0)
    node_names = default_collate(node_name_list)
    assert (len(gt_node_idx_list) == 0) or (len(gt_node_idx_list) == len(batch))
    assert (len(gt_edge_idx_list) == 0) or (len(gt_edge_idx_list) == len(batch))

    return (item_batch, node_names, node_categories_tensor, edge_categories_tensor,
            edge_connections_tensor, graph_idx_of_node_tensor, graph_idx_of_edge_tensor,
            gt_node_idx_list, gt_edge_idx_list, n_nodes_per_graph, n_edges_per_graph)
