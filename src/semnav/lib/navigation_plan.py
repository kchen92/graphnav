from __future__ import print_function

import numpy as np

from collections import namedtuple, OrderedDict

from semnav_ros.msg import NavCommandGoal
from semnav.lib.categories import NavPlanDifficulty


NavigationGoal = namedtuple('NavigationGoal', ['start_node', 'end_node'])


class NavigationPlan(object):
    """Navigation plan as a sequence of nodes/behaviors.
    """

    def __init__(self, node_list):
        assert len(node_list) > 0

        # Build edge list
        edge_list = []
        node2edge = OrderedDict()
        for idx, cur_node in enumerate(node_list):
            if (idx + 1) < len(node_list):
                next_node = node_list[idx + 1]
                for edge in cur_node.outgoing_edges:
                    if edge.end_node is next_node:
                        edge_list.append(edge)
                        node2edge[cur_node] = edge
                        break
        assert len(edge_list) == (len(node_list) - 1)

        self._node_list = tuple(node_list)
        self._edge_list = tuple(edge_list)
        self._node2edge = node2edge
        self._nav_goal = NavigationGoal(self.node_list[0], self.node_list[-1])
        assert self.is_valid_nav_plan() is True
        self._estimated_distance = self.compute_estimated_distance()  # Map coord units (meters)
        if len(node_list) < 10:
            self._difficulty = NavPlanDifficulty.easy
        elif len(node_list) < 20:
            self._difficulty = NavPlanDifficulty.moderate
        else:
            self._difficulty = NavPlanDifficulty.hard

    def compute_estimated_distance(self):
        total_dist = 0.
        for edge in self.edge_list:
            cur_dist = np.linalg.norm(edge.end_node.map_coord - edge.start_node.map_coord)
            total_dist += cur_dist
        return total_dist

    def percentage_plan_completed(self, last_valid_node):
        cur_node_step = 0
        for node in self.node_list:
            cur_node_step += 1
            if node is last_valid_node:
                break
        return float(cur_node_step) / len(self.node_list)

    def to_ros_msg(self, episode_idx):
        nav_plan_string = ' '.join([node.name for node in self.node_list])
        nav_command_goal = NavCommandGoal(episode_idx=episode_idx, nav_plan=nav_plan_string)
        return nav_command_goal

    @staticmethod
    def from_msg(sem_graph, ros_msg):
        """Decode a nav_plan_msg (ROS msg) to a NavigationPlan.
        """
        node_names = ros_msg.nav_plan.split(' ')
        node_list = [sem_graph.nodes[node_name] for node_name in node_names]
        nav_plan = NavigationPlan(node_list)
        return int(ros_msg.episode_idx), nav_plan

    def is_valid_nav_plan(self):
        """Performs the following checks to make sure the navigation plan is valid:
            - Navigation plan has length greater than 0
            - Navigation plan starts in room
            - Navigation plan ends in room

        We should never visit the same node twice for any given navigation plan.
        """
        return len(self.node_list) > 0

    @property
    def node_list(self):
        return self._node_list

    @property
    def edge_list(self):
        return self._edge_list

    @property
    def node2edge(self):
        return self._node2edge

    @property
    def nav_goal(self):
        return self._nav_goal

    @property
    def estimated_distance(self):
        return self._estimated_distance

    @property
    def difficulty(self):
        return self._difficulty
