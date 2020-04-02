#!/usr/bin/env python

from __future__ import print_function

import actionlib
import networkx as nx
import numpy as np
import pickle
import random
import rospy

from semnav.lib.image_map_transformer import Image2MapTransformer
from semnav.lib.navigation_brain import EpisodeResult
from semnav.lib.navigation_plan import NavigationGoal, NavigationPlan
from semnav.lib.sem_graph import SemGraph
from semnav.nodes.room_data_collector import RoomDataCollector
from semnav_ros.msg import NavCommandAction


class NavigationPlanner(RoomDataCollector):
    """Navigation planner.

    Creates navigation plans and sends them to the navigation brain, which executes the plan.
    """

    def __init__(self, yaml_filepath, start_navigator=True, sample_rooms_only=False, nav_plan_file=None):
        super(NavigationPlanner, self).__init__(start_navigator=start_navigator,
                                                sample_rooms_only=sample_rooms_only)
        self.sem_graph = SemGraph(yaml_filepath)
        self.nav_plan = None
        self.img2map = Image2MapTransformer(yaml_filepath)
        self.percentage_plan_completed_num = 0.
        self.percentage_plan_completed_denom = 0.

        if nav_plan_file is not None:
            print('Loading trajectories from: {}'.format(nav_plan_file))
            self.traj_idx = 0
            with open(nav_plan_file, 'rb') as f:
                self.trajectory_list = pickle.load(f)
        else:
            self.trajectory_list = None

    def start_navigator(self):
        self.client = actionlib.SimpleActionClient('navigation_command', NavCommandAction)
        self.client.wait_for_server()

    def get_plan_from_goal(self, nav_goal):
        """Return a NavigationPlan for the given NavigationGoal.
        """
        # Default is to get the shortest path
        node_list = self.sem_graph.find_shortest_path(nav_goal.start_node,
                                                      nav_goal.end_node)
        nav_plan = NavigationPlan(node_list=node_list)
        return nav_plan

    def get_start_node_with_shortest_path(self, start_node_1_name, start_node_2_name, end_node):
        shortest_path_1 = float('inf')
        shortest_path_2 = float('inf')
        if start_node_1_name in self.sem_graph.nodes:
            shortest_path_1 = nx.shortest_path_length(self.sem_graph.nxG, source=self.sem_graph.nodes[start_node_1_name], target=end_node)
        if start_node_2_name in self.sem_graph.nodes:
            shortest_path_2 = nx.shortest_path_length(self.sem_graph.nxG, source=self.sem_graph.nodes[start_node_2_name], target=end_node)
        if shortest_path_1 < shortest_path_2:
            chosen_start_node_name = start_node_1_name
        else:
            chosen_start_node_name = start_node_2_name
        start_node = self.sem_graph.nodes[chosen_start_node_name]
        return start_node

    def get_end_node_with_shortest_path(self, start_node, end_node_1_name, end_node_2_name):
        shortest_path_1 = float('inf')
        shortest_path_2 = float('inf')
        if end_node_1_name in self.sem_graph.nodes:
            shortest_path_1 = nx.shortest_path_length(self.sem_graph.nxG, source=start_node, target=self.sem_graph.nodes[end_node_1_name])
        if end_node_2_name in self.sem_graph.nodes:
            shortest_path_2 = nx.shortest_path_length(self.sem_graph.nxG, source=start_node, target=self.sem_graph.nodes[end_node_2_name])
        if shortest_path_1 < shortest_path_2:
            chosen_end_node_name = end_node_1_name
        else:
            chosen_end_node_name = end_node_2_name
        end_node = self.sem_graph.nodes[chosen_end_node_name]
        return end_node

    def choose_random_start_node(self, start_node, dir_1, dir_2):
        option_1 = start_node.name[:-1] + dir_1
        option_2 = start_node.name[:-1] + dir_2
        if (option_1 in self.sem_graph.nodes) and (option_2 in self.sem_graph.nodes):
            chosen_start_node_name = random.choice([option_1, option_2])
            start_node = self.sem_graph.nodes[chosen_start_node_name]
        return start_node

    def sample_trajectory(self):
        """Determine a NavigationGoal and a NavigationPlan.
        """
        if self.trajectory_list is None:  # Sample a random navigation plan
            cur_traj = super(NavigationPlanner, self).sample_trajectory()
        else:  # Execute trajectories from the trajectory list in order
            # If we are done evaluating all trajectories, print statistics and exit
            if self.traj_idx >= len(self.trajectory_list):
                print('--> Done running all trajectories!')
                raise rospy.ROSInterruptException  # Not the most graceful way to exit...

            # Print progress
            if (self.traj_idx + 1) % 1 == 0:
                print('--> Executing plan {} / {}'.format(self.traj_idx + 1, len(self.trajectory_list)))

            # Get the trajectory to execute
            cur_traj = self.trajectory_list[self.traj_idx]['traj']

        # Convert from pixel coordinates to xy coordinates
        start_pt_xy = self.img2map.pixel2map(cur_traj['start_pt'])
        end_pt_xy = self.img2map.pixel2map(cur_traj['end_pt'])

        # Get nodes
        if self.trajectory_list is None:
            start_room_name = self.img2map.get_room_name(start_pt_xy, xy=True)
            end_room_name = self.img2map.get_room_name(end_pt_xy, xy=True)
            if start_room_name in self.sem_graph.nodes:
                start_node = self.sem_graph.nodes[start_room_name]
            else:
                start_node = self.sem_graph.get_closest_node(start_room_name, start_pt_xy,
                                                             orientation=None, alignment_matters=False)
                if start_node.name.endswith('_door') and (start_node.name[:-5] in self.sem_graph.nodes):
                    start_node = self.sem_graph.nodes[start_node.name[:-5]]
                assert 'door' not in start_node.name

                if start_node.name[-2].isdigit() and (start_node.name.endswith('l') or start_node.name.endswith('r')):
                    start_node = self.choose_random_start_node(start_node, dir_1='l', dir_2='r')
                elif start_node.name[-2].isdigit() and (start_node.name.endswith('u') or start_node.name.endswith('d')):
                    start_node = self.choose_random_start_node(start_node, dir_1='u', dir_2='d')
            if end_room_name in self.sem_graph.nodes:
                end_node = self.sem_graph.nodes[end_room_name]
            else:
                end_node = self.sem_graph.get_closest_node(end_room_name, end_pt_xy,
                                                           orientation=None, alignment_matters=False)
                if end_node.name.endswith('_door') and (end_node.name[:-5] in self.sem_graph.nodes):
                    end_node = self.sem_graph.nodes[end_node.name[:-5]]
                assert 'door' not in end_node.name

                if end_node.name[-2].isdigit() and (end_node.name.endswith('l') or end_node.name.endswith('r')):
                    end_node_1_name = end_node.name[:-1] + 'l'
                    end_node_2_name = end_node.name[:-1] + 'r'
                    end_node = self.get_end_node_with_shortest_path(start_node, end_node_1_name, end_node_2_name)
                elif end_node.name[-2].isdigit() and (end_node.name.endswith('u') or end_node.name.endswith('d')):
                    end_node_1_name = end_node.name[:-1] + 'u'
                    end_node_2_name = end_node.name[:-1] + 'd'
                    end_node = self.get_end_node_with_shortest_path(start_node, end_node_1_name, end_node_2_name)

            if start_node.name[-2].isdigit() and (start_node.name.endswith('l') or start_node.name.endswith('r')):
                start_node_1_name = start_node.name[:-1] + 'l'
                start_node_2_name = start_node.name[:-1] + 'r'
                start_node = self.get_start_node_with_shortest_path(start_node_1_name, start_node_2_name, end_node)
            elif start_node.name[-2].isdigit() and (start_node.name.endswith('u') or start_node.name.endswith('d')):
                start_node_1_name = start_node.name[:-1] + 'u'
                start_node_2_name = start_node.name[:-1] + 'd'
                start_node = self.get_start_node_with_shortest_path(start_node_1_name, start_node_2_name, end_node)

            if end_node.name[-2].isdigit() and (end_node.name.endswith('l') or end_node.name.endswith('r')):
                end_node_1_name = end_node.name[:-1] + 'l'
                end_node_2_name = end_node.name[:-1] + 'r'
                end_node = self.get_end_node_with_shortest_path(start_node, end_node_1_name, end_node_2_name)
            elif end_node.name[-2].isdigit() and (end_node.name.endswith('u') or end_node.name.endswith('d')):
                end_node_1_name = end_node.name[:-1] + 'u'
                end_node_2_name = end_node.name[:-1] + 'd'
                end_node = self.get_end_node_with_shortest_path(start_node, end_node_1_name, end_node_2_name)
        else:
            start_node = self.sem_graph.nodes[self.trajectory_list[self.traj_idx]['start_node_name']]
            end_node = self.sem_graph.nodes[self.trajectory_list[self.traj_idx]['end_node_name']]

        nav_goal = NavigationGoal(start_node, end_node)
        self.nav_plan = self.get_plan_from_goal(nav_goal)
        for node in self.nav_plan.node_list[2:]:
            try:
                assert 'door' not in node.name, [node.name for node in self.nav_plan.node_list]
            except AssertionError:
                pass
        print('Estimated plan length (meters):', self.nav_plan.estimated_distance)

        # Compute orientation for self.compute_orientation()...a bit ugly
        if self.trajectory_list is None:
            start_node = self.nav_plan.node_list[0]
            edge = start_node.get_edge_to_node(self.nav_plan.node_list[1])
            start_orientation = edge.angle
            noise_degrees = 20.
            noise_radians = float(noise_degrees) / 180 * np.pi
            delta = np.random.uniform(low=-noise_radians, high=noise_radians)
            start_orientation += delta
            self.start_orientation = start_orientation
        else:
            self.start_orientation = self.trajectory_list[self.traj_idx]['start_orientation']
            self.traj_idx += 1  # Increment index in trajectory list

        # Set self.start_node and self.end_node for NavPlanFileCreator
        self.start_node = start_node
        self.end_node = end_node

        return cur_traj

    def compute_orientation(self, cur_traj):
        return self.start_orientation  # Ugly

    def send_goal(self, end_pt, end_orientation):
        """Send the NavigationPlan to the NavigationBrain.
        """
        print('Sending goal: {} to {}'.format(self.nav_plan.nav_goal.start_node.name,
                                              self.nav_plan.nav_goal.end_node.name))
        nav_plan_msg = self.nav_plan.to_ros_msg(self.episode_count)

        self.client.wait_for_server()
        self.client.send_goal(nav_plan_msg)
        self.client.wait_for_result()
        result_msg = self.client.get_result()
        result = EpisodeResult(result_msg.episode_result)

        self.percentage_plan_completed_num += result_msg.percentage_plan_completed
        self.percentage_plan_completed_denom += 1.
        print('')

        return result is EpisodeResult.SUCCESS


if __name__ == '__main__':
    nav_plan_file = None
    sample_rooms_only = False

    try:
        yaml_filepath = rospy.get_param("/map_file")
        print('Using yaml filepath: {}'.format(yaml_filepath))
        navigation_planner = NavigationPlanner(yaml_filepath, sample_rooms_only=sample_rooms_only,
                                               nav_plan_file=nav_plan_file)
        navigation_planner.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation planner terminated.")
