from __future__ import print_function
from __future__ import division

import torch.nn as nn

from semnav.lib.graph_net_blocks import GNBlock, CNNBlock
from semnav.lib.categories import BehaviorCategory, RoomCategory


class GraphNet(nn.Module):

    def __init__(
            self,
            img_in_channels,
            num_layers,
            aggregate_method,
            in_global_feat_size,
            in_node_feat_size,
            in_edge_feat_size,
            intermediate_global_feat_size,
            intermediate_node_feat_size,
            intermediate_edge_feat_size,
            out_global_feat_size=None,
            out_node_feat_size=None,
            out_edge_feat_size=None,
            ):
        """Construct a Graph Network, which is consisted of a CNN and a series of graph network
        (GNBlock) layers.

        Args:
            img_in_channels: Number of input channels to the CNN.
            num_layers: Number of graph network layers.
            in_global_feat_size: Output dimension of CNN, which is the input dimension for the
                    graph global feature to the graph network.
            in_node_feat_size: Dimension of the input features for each node. This is also the
                    dimension of the node embedding lookup table.
            in_edge_feat_size: Dimension of the input features for each edge. This is also the
                    dimension of the edge embedding lookup table.
            intermediate_global_feat_size: Dimension of global features for intermediate graph
                    network layers.
            intermediate_node_feat_size: Dimension of node features for intermediate graph
                    network layers.
            intermediate_edge_feat_size: Dimension of edge features for intermediate graph
                    network layers.
            out_global_feat_size: Dimension of global features output from the last graph network layer.
            out_node_feat_size: Dimension of node features output from the last graph network layer.
            out_edge_feat_size: Dimension of edge features output from the last graph network layer.
        """
        super(GraphNet, self).__init__()
        self.is_recurrent = False
        self.num_layers = num_layers
        if out_node_feat_size is None:
            out_node_feat_size = 1
        if out_edge_feat_size is None:
            out_edge_feat_size = 1

        self.cnn_encoder = CNNBlock(in_channels=img_in_channels, out_dim=in_global_feat_size)
        self.node_embeddings = nn.Embedding(num_embeddings=len(RoomCategory),
                                            embedding_dim=in_node_feat_size)
        self.edge_embeddings = nn.Embedding(num_embeddings=len(BehaviorCategory),
                                            embedding_dim=in_edge_feat_size)

        self.gn_layers = self._build_graph_net_layers(
            num_layers,
            in_global_feat_size,
            in_node_feat_size,
            in_edge_feat_size,
            intermediate_global_feat_size,
            intermediate_node_feat_size,
            intermediate_edge_feat_size,
            out_global_feat_size,
            out_node_feat_size,
            out_edge_feat_size,
            aggregate_method,
            )

    @staticmethod
    def _build_graph_net_layers(
            num_layers,
            in_global_feat_size,
            in_node_feat_size,
            in_edge_feat_size,
            intermediate_global_feat_size,
            intermediate_node_feat_size,
            intermediate_edge_feat_size,
            out_global_feat_size,
            out_node_feat_size,
            out_edge_feat_size,
            aggregate_method,
            ):
        gn_layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            cur_in_global_feat_size = intermediate_global_feat_size
            cur_in_node_feat_size = intermediate_node_feat_size
            cur_in_edge_feat_size = intermediate_edge_feat_size
            cur_out_global_feat_size = intermediate_global_feat_size
            cur_out_node_feat_size = intermediate_node_feat_size
            cur_out_edge_feat_size = intermediate_edge_feat_size
            if layer_idx == 0:  # First layer
                cur_in_global_feat_size = in_global_feat_size
                cur_in_node_feat_size = in_node_feat_size
                cur_in_edge_feat_size = in_edge_feat_size
            elif layer_idx == (num_layers - 1):  # Last layer
                cur_out_global_feat_size = out_global_feat_size
                cur_out_node_feat_size = out_node_feat_size
                cur_out_edge_feat_size = out_edge_feat_size
            gn_block = GNBlock(cur_in_global_feat_size, cur_in_node_feat_size,
                               cur_in_edge_feat_size,
                               out_global_feat_size=cur_out_global_feat_size,
                               out_node_feat_size=cur_out_node_feat_size,
                               out_edge_feat_size=cur_out_edge_feat_size,
                               aggregate_method=aggregate_method)
            gn_layers.append(gn_block)
        return gn_layers

    def forward(self, image_stack, input_dict):
        """Given the current subgraph and the input images, predict the node/edge in the subgraph.

        Steps:
        1. Forward pass through the CNN to produce a global feature vector.
        2. Perform the forward passes through the graph net layers.

        Args:
            subgraph_batch: Subgraph batch.
            image_stack: Input image stack [B, C, H, W] where B is the batch size and C is the
                    number of channels.

        Returns:
            scores: Score for each node/edge in the graph.
        """
        # Forward pass through CNN encoder
        initial_global_feature_batch = self.cnn_encoder(image_stack)  # batch_size x global_feat_dim

        global_features = initial_global_feature_batch
        node_features = self.node_embeddings(input_dict['node_categories'])
        edge_features = self.edge_embeddings(input_dict['edge_categories'])
        for layer in self.gn_layers:
            global_features, node_features, edge_features = layer(
                global_features,
                node_features,
                edge_features,
                input_dict['edge_connections'],
                input_dict['graph_idx_of_node'],
                input_dict['graph_idx_of_edge'],
                )
        return global_features, node_features, edge_features
