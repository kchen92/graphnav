"""Visualize the dataset by making it into a video.
"""

from __future__ import print_function
from __future__ import division

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import torch

from semnav.config import get_config
from semnav.dataset import get_split_datasets, load_dataset, load_dataset_splits

from semnav.dataset.graph_net_frame_dataset import GraphNetFrameDataset
from semnav.dataset.graph_net_dataset import graph_net_collate_fn
from semnav.dataset.trajectory_dataset import TrajectoryDataset
from semnav.lib.categories import SemanticCategory
from semnav.lib.image_map_transformer import Image2MapTransformer
from semnav.lib.sem_graph import SemGraph
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.utils import make_grid


class DatasetVisualizer(object):

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self):
        self.cfg = get_config()
        self.fig = plt.figure()
        self.softmax = torch.nn.Softmax(dim=0)

        # Graph visualizer attributes
        self.ax = self.fig.gca()

        self.img_write_idx = 0

    def visualize_sample(self, sample, top_right=None):
        """Visualize a frame data.

        Args:
            sample: Parsed frame data.
            top_right: String to display in the upper right corner.

        Returns:
            to_break: Whether or not to break out of the data visualizer main loop.
        """
        if isinstance(sample, tuple):  # Graph net dataset
            (sample, node_names, node_categories_tensor, edge_categories_tensor,
             edge_connections_tensor, graph_idx_of_node_tensor, graph_idx_of_edge_tensor,
             gt_node_idx_list, gt_edge_idx_list, n_nodes_per_graph, n_edges_per_graph) = sample
            has_graph_info = True
            area_name = sample.get('area_name')
            yaml_filepath = os.path.join(self.cfg.maps_root, area_name[0] + '.yaml')
            self.sem_graph = SemGraph(yaml_filepath)
            self.img2map = Image2MapTransformer(yaml_filepath)
        else:
            has_graph_info = False

        rgb_img = sample['rgb'][0]
        depth_img = torch.cat([sample['depth'][0]] * 3, dim=0) / self.cfg.max_depth
        images = [rgb_img, depth_img]
        semantic_img = sample['semantics_rgb'][0]  # Semantic image in rgb form
        if torch.sum(semantic_img) > 0:
            semantic_img = semantic_img.type(torch.FloatTensor) / 255.
            semantic_img = semantic_img.permute(2, 0, 1)
            images.append(semantic_img)

            # Visualize binary mask of door
            semantic_idx = sample['semantics_idx'][0]  # Semantic image in rgb form
            door_mask = semantic_idx == SemanticCategory.door
            door_mask = door_mask.type(torch.FloatTensor)
            images.append(door_mask)
        pred_str = sample.get('prediction_str')
        if has_graph_info is True:
            # Re-create subgraph
            nxG_compact = nx.OrderedDiGraph()
            pos_dic = dict()

            # Create nodes
            for node_name in node_names:
                # Create node
                nxG_compact.add_node(node_name)
                y, x = self.img2map.map2pixel(self.sem_graph.nodes[node_name].map_coord)
                pos_dic[node_name] = (x, y)

            # Create edges
            for u, v in edge_connections_tensor.tolist():
                nxG_compact.add_edge(node_names[u], node_names[v])

            if (pred_str is not None) and (self.cfg.behaviornet_type == 'graph_net'):
                net_output, pred_str = pred_str
                global_features, node_features, edge_features = net_output
                node_colors = []
                for node_name in nxG_compact.nodes:
                    if node_name == pred_str.split(' ')[0]:
                        node_colors.append('r')
                    elif node_name == sample.get('localized_node_name')[0]:
                        node_colors.append('g')
                    else:
                        node_colors.append('#A0CBE2')

                # Normalize predictions
                edge_features_normalized = self.softmax(edge_features)
                edge_colors = [w for w in torch.squeeze(edge_features_normalized).tolist()]
                nx.draw_networkx(nxG_compact, pos=pos_dic, with_labels=False, node_size=20, node_color=node_colors,
                                 edge_color=edge_colors, width=2., edge_cmap=plt.cm.Blues, ax=self.ax)
            else:
                subgraph_nodes = [n for n in nxG_compact.nodes]
                nx.draw_networkx_nodes(nxG_compact, pos=pos_dic, nodelist=subgraph_nodes, node_color='#759aad', ax=self.ax)
                nx.draw_networkx_nodes(nxG_compact, pos=pos_dic, nodelist=[sample.get('localized_node_name')[0]], node_color='r', ax=self.ax)
                nx.draw_networkx_edges(nxG_compact, pos=pos_dic, ax=self.ax)
                nx.draw_networkx_labels(nxG_compact, pos=pos_dic, font_family='Gentium Book Basic')

            map_img = cv2.imread(os.path.join(self.cfg.maps_root, area_name[0] + '.png'))
            self.ax.imshow(map_img)
            self.ax.set_xlim(0, map_img.shape[1])
            self.ax.set_ylim(0, map_img.shape[0])
            self.ax.invert_yaxis()
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # This need to be set to not have any offset in the graph

            # Make the longest edge 10 inches
            max_edge_len_inches = 10.
            longest_edge_in_px = max(map_img.shape[0], map_img.shape[1])
            w_inches = float(map_img.shape[1]) / longest_edge_in_px * max_edge_len_inches
            h_inches = float(map_img.shape[0]) / longest_edge_in_px * max_edge_len_inches
            self.fig.set_size_inches(w_inches, h_inches)
            self.fig.canvas.draw()

            graph_img_np = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            graph_img_np = graph_img_np.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        grid_img = make_grid(images)

        # Parse sample
        behavior_id = sample.get('behavior_id')
        localized_behavior_id = sample.get('localized_behavior_id')
        affordance_vec = sample.get('affordance_vec')
        node_name = sample.get('node_name')
        phase = sample.get('phase')
        room_name = sample['room_name']
        is_invalid = sample['is_invalid'][0]
        sequence_idx = sample['sequence_idx'].numpy()[0]

        img_np = grid_img.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
        img_uint = img_np * 255.
        img_uint = img_uint[:, :, ::-1]
        img_uint = img_uint.astype(np.uint8).copy()
        cv2.putText(img_uint, text=room_name[0], org=(350, 220), fontFace=self.FONT, fontScale=1,
                    color=(255, 0, 0), thickness=2)
        cv2.putText(img_uint, text=str(sequence_idx), org=(10, 30), fontFace=self.FONT, fontScale=1,
                    color=(255, 0, 0), thickness=2)
        if top_right is not None:
            assert isinstance(top_right, str)
            cv2.putText(img_uint, text=top_right, org=(350, 30), fontFace=self.FONT, fontScale=1,
                        color=(255, 0, 0), thickness=2)
        if behavior_id is not None:
            cv2.putText(img_uint, text=behavior_id[0], org=(10, 220), fontFace=self.FONT, fontScale=1,
                        color=(0, 0, 255), thickness=2)
        if localized_behavior_id is not None:
            cv2.putText(img_uint, text=localized_behavior_id[0], org=(10, 150), fontFace=self.FONT, fontScale=1,
                        color=(0, 255, 255), thickness=2)
            cv2.putText(img_uint, text=sample.get('localized_node_name')[0], org=(10, 180), fontFace=self.FONT, fontScale=1,
                        color=(0, 255, 255), thickness=2)
        if affordance_vec is not None:
            affordance_list = TrajectoryDataset.affordance_vec2list(affordance_vec.numpy()[0])
            if len(affordance_list) > 0:
                cur_affordances = ' '.join([x for x in affordance_list])
                cv2.putText(img_uint, text=cur_affordances, org=(350, 30), fontFace=self.FONT, fontScale=1,
                            color=(255, 0, 0), thickness=2)
        if node_name is not None:
            cv2.putText(img_uint, text=node_name[0], org=(350, 60), fontFace=self.FONT, fontScale=1,
                        color=(255, 0, 0), thickness=2)
        if phase is not None:
            phase = np.round(phase.numpy()[0], decimals=2)
            cv2.putText(img_uint, text=str(phase), org=(350, 90), fontFace=self.FONT, fontScale=1,
                        color=(255, 0, 0), thickness=2)
        if pred_str is not None:
            cv2.putText(img_uint, text=str(pred_str), org=(10, 80), fontFace=self.FONT, fontScale=0.75,
                        color=(255, 0, 0), thickness=2)
        if is_invalid == 1:
            cv2.putText(img_uint, text='invalid', org=(10, 100), fontFace=self.FONT, fontScale=1,
                        color=(0, 0, 255), thickness=2)
        cv2.imshow('image', img_uint)
        if has_graph_info is True:
            cv2.imshow("Graph", cv2.cvtColor(graph_img_np, cv2.COLOR_BGR2RGB))
            self.ax.clear()
        k = cv2.waitKey(0)

        # Write images
        for modality_idx, img in enumerate(images):
            img_np = img.numpy()
            img_np = np.transpose(img_np, (1, 2, 0))
            img_uint = img_np * 255.
            img_uint = img_uint[:, :, ::-1]

        # Save depth map
        self.img_write_idx += 1

        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()
            return True
        else:
            return False

    def visualize_data_loader_item(self, dataset_item, use_frame_by_frame=False):
        if (use_frame_by_frame is True) or (self.cfg.dataset_type == 'frame_by_frame') or (self.cfg.dataset_type == 'single_frame_graph_net'):
            to_break = self.visualize_sample(dataset_item)
        elif self.cfg.dataset_type == 'temporal':
            assert len(dataset_item) == self.cfg.n_frames_per_sample
            for i in range(len(dataset_item)):
                to_break = self.visualize_sample(dataset_item[i], top_right=str(i))

                if to_break:
                    break
        else:
            raise ValueError('Dataset type missing or not supported.')
        return to_break

    def run(self, bag_dir_name=None):
        """Visualize data in a Dataset.

        Usage:
            - Press any key to move onto the next frame.
            - Press ESC to terminate the program.
        """
        print('Loading dataset')
        if self.cfg.dataset is not None:
            train_set, val_set, test_set = get_split_datasets(self.cfg.dataset)
            self.dataset = ConcatDataset([train_set, val_set, test_set])
        else:
            if os.path.isdir(os.path.join(self.cfg.dataset_root, bag_dir_name, 'train')):
                train_set, val_set, test_set = load_dataset_splits(bag_dir_name, self.cfg)
                self.dataset = ConcatDataset([train_set, val_set, test_set])
            else:
                self.dataset = load_dataset(os.path.join(self.cfg.dataset_root, bag_dir_name), self.cfg)
        if (self.cfg.dataset_type == 'graph_net'):
            dataset_loader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=False, num_workers=0,
                                        collate_fn=graph_net_collate_fn)
        elif (self.cfg.dataset_type == 'single_frame_graph_net'):
            dataset_loader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=False, num_workers=0,
                                        collate_fn=GraphNetFrameDataset.collate_fn)
        else:
            dataset_loader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=False, num_workers=0)
        print('Done!')

        for idx, dataset_item in enumerate(dataset_loader):
            to_break = self.visualize_data_loader_item(dataset_item)
            if to_break:
                break


if __name__ == '__main__':
    # Manually edit this :)
    bag_dir_name = 'path_to_dir'
    dataset_visualizer = DatasetVisualizer()
    dataset_visualizer.run(bag_dir_name=bag_dir_name)
