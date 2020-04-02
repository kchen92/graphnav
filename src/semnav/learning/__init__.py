from __future__ import print_function
from __future__ import division

import torch

from semnav.learning.behavior_net.behavior_cnn import BehaviorCNN
from semnav.learning.behavior_net.behavior_rnn import BehaviorRNN
from semnav.learning.graph_net.graph_net import GraphNet
from semnav.lib.categories import SemanticCategory


def get_net(net_class, cfg):
    if cfg.dataset_type == 'frame_by_frame':
        in_channels = 1
    elif ((cfg.dataset_type == 'temporal') or (cfg.dataset_type == 'phase_net')
          or (cfg.dataset_type == 'graph_net')):
        in_channels = cfg.n_frames_per_sample
    else:
        raise ValueError('Unsupported dataset type.')

    if cfg.use_semantic_class is not None:
        in_channels *= 2  # For every timestep, we have 1 more frame/channel

    if net_class == 'behavior_cnn':
        net = BehaviorCNN(in_channels=in_channels).to(cfg.device)
    elif net_class == 'behavior_rnn':
        net = BehaviorRNN(rnn_type='lstm', hidden_size=512, num_layers=2).to(cfg.device)
    elif net_class == 'graph_net':
        # Because of GNBlock execution order (edge -> node -> global), if doing node classification,
        # Make sure out_edge_feat_size is of decent size (e.g. 256 instead of 1) since the NodeBlock
        # utilizes the updated edge features
        out_global_feat_size = None
        out_node_feat_size = 1
        out_edge_feat_size = 1

        if (cfg.gn_classification == 'node') or (cfg.gn_classification == 'joint'):
            assert out_node_feat_size == 1
        if (cfg.gn_classification == 'edge') or (cfg.gn_classification == 'joint'):
            assert out_edge_feat_size == 1
        net = GraphNet(img_in_channels=in_channels, num_layers=cfg.n_graphnet_layers,
                       aggregate_method=cfg.aggregate_method, in_global_feat_size=512,
                       in_node_feat_size=cfg.graphnet_feat_size,
                       in_edge_feat_size=cfg.graphnet_feat_size,
                       intermediate_global_feat_size=cfg.graphnet_feat_size,
                       intermediate_node_feat_size=cfg.graphnet_feat_size,
                       intermediate_edge_feat_size=cfg.graphnet_feat_size,
                       out_global_feat_size=out_global_feat_size,
                       out_node_feat_size=out_node_feat_size,
                       out_edge_feat_size=out_edge_feat_size).to(cfg.device)
    else:
        raise ValueError('Unsupported behavior network type.')
    print(net)
    print('')

    if cfg.ckpt_path is not None:
        print('Loading checkpoint:', cfg.ckpt_path)
        net.load_state_dict(torch.load(cfg.ckpt_path))
    else:
        print('Using random weight initialization.')
    print('')

    return net


def decode_temporal(batch, device, key, add_batch_dim):
    """Get the last frame as the target output.

    Args:
        batch: Batch of data.
        device: Device (e.g. GPU or CPU).
        key: Key for the TrajectoryDataset frame dict.
        add_batch_dim: Boolean for whether to add a batch dim.
    """
    target_output = batch[-1][key]
    if not isinstance(target_output, list):
        target_output = target_output.to(device)
    if add_batch_dim is True:
        target_output = torch.unsqueeze(target_output, dim=0).to(device)
    return target_output


def get_frame_data(batch, device, decode_type, key):
    """Get a single_frame input data or temporal stack of input data.
    """
    # Set up the depth input
    if decode_type.startswith('single_frame'):  # Set up input for single-frame depth
        depth = batch[key].to(device)
        if depth.dim() == 3:  # Add batch dimension if it doesn't exist
            depth = torch.unsqueeze(depth, dim=0).to(device)
    elif decode_type.startswith('temporal'):  # Set up input for depth stack (temporal)
        depth_list = [x[key] for x in batch]  # Most recent image is at the end
        if depth_list[0].dim() == 3:  # Add batch dimension if it doesn't exist
            depth_list = [torch.unsqueeze(depth, dim=0) for depth in depth_list]
        depth_stack = torch.cat(depth_list, dim=1)
        depth = depth_stack.to(device)
    else:
        raise ValueError('Unsupported decode type.')
    return depth


def merge_input_tensors(input_tensors):
    # Combine into a tensor like:
    # [depth channel 1, semantics channel 1, depth channel 2, semantics channel 2, etc.]
    n_frames = input_tensors[0].shape[1]
    tensor_list = []
    for frame_idx in range(n_frames):
        for cur_input_tensor in input_tensors:
            tensor_list.append(torch.unsqueeze(cur_input_tensor[:, frame_idx], dim=1))
    input_tensor = torch.cat(tensor_list, dim=1)
    return input_tensor


def decode_batch(batch, decode_type, cfg):
    """Decode the batch.

    Args:
        batch: Batch of data returned from the DataLoader.
        decode_type: 'single_frame' or 'temporal' denoting how to decode the batch.
        device: Device.

    Returns:
        vel: If decode_type is 'single_frame', this is shape (batch_size x 2).
        depth: If decode_type is 'single_frame', this is shape (batch_size x 1 x 240 x 320).
    """
    if cfg.dataset_type == 'graph_net':
        if not isinstance(batch, dict):
            has_graph_info = True
            (batch, node_names, node_categories, edge_categories, edge_connections, graph_idx_of_node,
             graph_idx_of_edge, gt_node_idx, gt_edge_idx, n_nodes_per_graph, n_edges_per_graph) = batch
        else:
            has_graph_info = False
    device = cfg.device

    # Default
    input_tensors = []

    # Figure out if we need to add batch dim
    add_batch_dim = False
    if decode_type.startswith('single_frame'):  # Set up input for single-frame depth
        if batch['depth'].dim() == 3:  # Add batch dimension if it doesn't exist
            add_batch_dim = True
    elif decode_type.startswith('temporal'):  # Set up input for depth stack (temporal)
        if batch[0]['depth'].dim() == 3:  # Add batch dimension if it doesn't exist
            add_batch_dim = True
    else:
        raise ValueError

    depth = get_frame_data(batch, device, decode_type, key='depth')

    # Set up the target output
    if decode_type == 'single_frame':
        vel = batch['vel'].to(device)
        if vel.dim() == 1:  # Add batch dimension if it doesn't exist
            vel = torch.unsqueeze(vel, dim=0).to(device)
    elif decode_type == 'temporal':
        # Velocity of the last frame is the target velocity
        vel = decode_temporal(batch, device, key='vel', add_batch_dim=add_batch_dim)
    elif decode_type == 'single_frame_phase':
        phase = batch['phase'].to(device)
        if (phase.dim() == 1):  # Add scalar dimension if it doesn't exist
            phase = torch.unsqueeze(phase, dim=1).to(device)
        vel = phase.type(torch.FloatTensor).to(device)
    elif (decode_type == 'temporal_graph_net') or (decode_type == 'single_frame_graph_net'):
        if has_graph_info is False:  # Special case that occurs in evaluation when we dont have GT labels for the current frame
            vel = None
        else:
            (batch, node_names, node_categories, edge_categories,
                edge_connections, graph_idx_of_node, graph_idx_of_edge)
            graph_net_input_dict = {
                'node_names': node_names,
                'node_categories': node_categories.to(device),
                'edge_categories': edge_categories.to(device),
                'edge_connections': edge_connections.to(device),
                'graph_idx_of_node': graph_idx_of_node.to(device),
                'graph_idx_of_edge': graph_idx_of_edge.to(device),
                }
            target_output = {
                'gt_node_idx': gt_node_idx,
                'gt_edge_idx': gt_edge_idx,
                'n_nodes_per_graph': n_nodes_per_graph,
                'n_edges_per_graph': n_edges_per_graph,
                'batch': batch,
                }
            vel = graph_net_input_dict, target_output
    else:
        raise ValueError('Unsupported dataset type.')

    # By default, use depth as input:
    input_tensors.append(depth)

    if cfg.use_semantic_class is not None:
        semantic_idx_tensor = get_frame_data(batch, device, decode_type, key='semantics_idx')
        door_mask = semantic_idx_tensor == SemanticCategory[cfg.use_semantic_class]
        door_mask = door_mask.type(torch.FloatTensor).to(device)
        input_tensors.append(door_mask)

    # Consolidate different input types into a single tensor
    if len(input_tensors) > 1:
        if decode_type.startswith('single_frame'):
            input_tensor = torch.cat(input_tensors, dim=1)
        else:  # Assumes each tensor has channel dim == 1
            input_tensor = merge_input_tensors(input_tensors)
    else:
        input_tensor = input_tensors[0]

    return vel, input_tensor
