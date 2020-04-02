from __future__ import print_function


import json
import networkx as nx
import numpy as np
import os
import yaml

from collections import namedtuple, Counter

from semnav.lib.image_map_transformer import Image2MapTransformer
from semnav.lib.behavior_detectors import BehaviorDetector
from semnav.lib.categories import NavPlanDifficulty
from semnav.lib.success_tracker import SuccessTracker
from semnav.lib.utils import compute_angle_delta, compute_dist_to_node
try:
    from tf.transformations import euler_from_quaternion
except ImportError:
    from semnav.third_party.transformations import euler_from_quaternion


Edge = namedtuple('Edge', ['angle', 'behavior_id', 'start_node', 'end_node', 'success_tracker'])


class Node(object):

    def __init__(self, name, map_coord, node_size=1.):
        """Constructor.

        Args:
            node_size: Size of node, AKA how distances scale relative to node. See
                    semnav.utils.compute_dist_to_node().
        """
        self.name = name
        self.map_coord = map_coord
        self.node_size = node_size
        self.success_tracker = SuccessTracker()
        self._outgoing_edges = []
        self._incoming_edges = []

    def add_outgoing_edge(self, edge):
        assert edge.start_node == self
        self._outgoing_edges.append(edge)

    def add_incoming_edge(self, edge):
        assert edge.end_node == self
        self._incoming_edges.append(edge)

    def get_edge_to_node(self, end_node):
        """Return the outgoing edge that goes to end_node.
        """
        for edge in self.outgoing_edges:
            if edge.end_node is end_node:
                return edge
        return None

    @property
    def edges(self):
        return self.outgoing_edges + self.incoming_edges

    @property
    def affordances(self):
        affordances = []
        for edge in self.outgoing_edges:
            affordances.append(edge.behavior_id)
        return tuple(affordances)

    def get_smallest_angle(self, orientation):
        """Compute the smallest (yaw) angle delta between a given orientation and any of the
        outgoing or incoming edges. Return the angle delta and the edge.

        Args:
            orientation: Orientation (e.g. cur_frame['gt_odom']['orientation']).

        Returns:
            min_edge: Edge with the smallest (yaw) angle delta.
            min_yaw_delta: Difference in angle between closest edge and input orientation.
        """
        _, _, cur_yaw = euler_from_quaternion(orientation)
        min_yaw_delta = float('inf')
        min_edge = None
        for edge in self.edges:
            angle_delta = abs(compute_angle_delta(cur_yaw, edge.angle))
            if angle_delta < min_yaw_delta:
                min_edge = edge
                min_yaw_delta = angle_delta
        return min_edge, min_yaw_delta

    @property
    def outgoing_edges(self):
        return self._outgoing_edges

    @property
    def incoming_edges(self):
        return self._incoming_edges


class SemGraph(object):
    """Abstraction for graph.
    """

    def __init__(self, yaml_filepath):
        img2map = Image2MapTransformer(yaml_filepath)

        # Load yaml file containing map data
        with open(yaml_filepath, 'r') as f:
            map_yaml = yaml.load(f)
        graph_filepath = os.path.join(os.path.dirname(yaml_filepath), map_yaml['graph_filepath'])
        node_sizes = map_yaml.get('node_sizes', {})
        self.nodes, self.edges, self.nxG = self.parse_graph(graph_filepath, img2map, node_sizes)

        # Use this primarily for the in_room() method
        self.behavior_detector = BehaviorDetector()

        # Keep track of statistics for each behavior
        behavior_id_set = set([edge.behavior_id for edge in self.edges])
        self.behavior_id_success_trackers = {behavior_id: SuccessTracker()
                                             for behavior_id in behavior_id_set}

        # Success rate for each difficulty
        self.difficulty_success_trackers = {difficulty: SuccessTracker()
                                            for difficulty in NavPlanDifficulty}

        # Percentage plan completed per difficulty
        self.difficulty_percentage_plan_complete = {difficulty: []
                                                    for difficulty in NavPlanDifficulty}

        self.behavior_preds_per_edge = {edge: Counter() for edge in self.edges}
        self.executed_nav_plans = []

    @staticmethod
    def compute_angle(start_node, end_node):
        """Compute the angle of the edge between two nodes.
        """
        delta_x = end_node.map_coord[0] - start_node.map_coord[0]
        delta_y = end_node.map_coord[1] - start_node.map_coord[1]
        return np.arctan2(delta_y, delta_x)

    @classmethod
    def parse_graph(cls, graph_filepath, img2map, node_sizes={}):
        """Parse the graph.

        raw_nodes: A list of nodes in the graph, where each node is represented as a list. The
        Example node:
        [495.0, 298.0, 501.0, 304.0, {u'tags': u'node_0',
                                      u'outline': u'red',
                                      u'fill': u'red'}]

        Example edge:
        [435.0, 301.0, 498.0, 301.0, {u'tags': u'edge_0 node_1 node_0',
                                      u'arrow': u'last',
                                      u'fill': u'green'}]

        Example node text:
        [437.0, 276.0, {u'text': u'WC_1',
                        u'fill': u'blue',
                        u'tags': u'node_1'}]

        Example edge text:
        [475.5, 268.0, {u'text': u'tr',
                        u'fill': u'blue',
                        u'tags': u'edge_333 node_183 node_1'}]
        """
        with open(graph_filepath, 'r') as f:
            graph_data = json.load(f)
        raw_nodes = graph_data['oval']
        raw_node_and_edge_text = graph_data['text']
        raw_edges = graph_data['line']

        # Tag is node_id or edge_tag, text is node name or edge behavior
        tag2text = {str(text[2]['tags']): str(text[2]['text']) for text in raw_node_and_edge_text}

        # Create a networkx directed graph
        nxG = nx.OrderedDiGraph()

        # Build node dict
        nodes = {}
        for raw_node in raw_nodes:
            node_id = raw_node[4]['tags']
            node_name = tag2text[node_id]

            # Get pixel coordinates [row, col]
            x = (raw_node[0] + raw_node[2]) / 2
            y = (raw_node[1] + raw_node[3]) / 2
            px_coord = np.asarray([[y, x]])  # 1x2 array
            map_coord = img2map.pixel2map(px_coord)  # Map coordinates as 1x2 array

            # Add node to node dict
            node_size = node_sizes.get(node_name, 1.)
            if node_name in node_sizes:
                print('Setting node size of {} to {}'.format(node_name, node_size))
            nodes[node_name] = Node(node_name, map_coord[0], node_size=node_size)

            nxG.add_node(nodes[node_name])

        # Build edge list and add to graph
        edges = []
        print('Check angle computation')
        for raw_edge in raw_edges:
            edge_id, start_node_id, end_node_id = [str(x) for x in raw_edge[4]['tags'].split(' ')]
            behavior_id = tag2text[raw_edge[4]['tags']]
            start_node_name = tag2text[start_node_id]
            end_node_name = tag2text[end_node_id]

            # Create edge
            start_node = nodes[start_node_name]
            end_node = nodes[end_node_name]
            angle = cls.compute_angle(start_node, end_node)
            edge = Edge(angle=angle, behavior_id=behavior_id, start_node=start_node, end_node=end_node, success_tracker=SuccessTracker())

            # Add edge to node edge lists
            start_node.add_outgoing_edge(edge)
            end_node.add_incoming_edge(edge)

            edges.append(edge)

            nxG.add_edge(start_node, end_node, object=edge)
        nx.freeze(nxG)

        return nodes, edges, nxG

    def get_closest_node(self, room_name, position, orientation, alignment_matters):
        """Get the closest node to the given position.

        Args:
            alignment_matters: Boolean for whether we care that the orientation matches edge angle.
        """
        closest_node = None
        min_dist = float('inf')
        for cur_node_name, cur_node in self.nodes.iteritems():
            # Flag for whether the room_name belongs to an actual room (e.g. office instead of corridor)
            in_room = self.behavior_detector.in_room_by_name(room_name)

            same_room_check = not in_room or (in_room and cur_node_name.startswith(room_name))
            cur_dist = compute_dist_to_node(position, cur_node)
            if (same_room_check  # If in room, make sure node is in room
                    and (not alignment_matters or (alignment_matters and (cur_node.get_smallest_angle(orientation)[1] < (0.20 * np.pi))))
                    and (cur_dist < min_dist)):
                closest_node = cur_node
                min_dist = cur_dist
        return closest_node

    def find_shortest_path_by_name(self, start_node_name, end_node_name):
        return self.find_shortest_path(self.nodes[start_node_name], self.nodes[end_node_name])

    def find_shortest_path(self, start_node, end_node):
        return nx.shortest_path(self.nxG, start_node, end_node)

    def find_all_paths_by_name(self, start_node_name, end_node_name):
        return self.find_all_paths_fast(self.nodes[start_node_name], self.nodes[end_node_name])

    def find_all_paths_fast(self, start_node, end_node):
        # This can be used once we find a way to not include paths which enter and and leave rooms
        raise NotImplementedError
        return nx.all_simple_paths(self.nxG, start_node, end_node)
