#!/usr/bin/env python

from __future__ import print_function

import copy
import rospy

from semnav.lib.categories import behavior_id2category
from semnav.learning import get_net
from semnav.learning.graph_net.graph_net_evaluator import GraphNetEvaluator
from semnav.lib.navigation_brain import NavigationBrain


class GraphNetNavigationBrain(NavigationBrain):
    """Use the GraphNet model to select which behaviors to execute at runtime.
    """

    def __init__(self, mode='sim'):
        """Constructor.

        Args:
            mode: 'real' or 'sim', it will change which topics to subscribe.
        """
        yaml_filepath = rospy.get_param("/map_file")
        print('Using yaml filepath: {}'.format(yaml_filepath))
        super(GraphNetNavigationBrain, self).__init__(node_name='graph_net_navigation_brain',
                                                      mode=mode, yaml_filepath=yaml_filepath)

        while self.last_depth is None:
            rospy.sleep(0.2)  # Let some messages come in

        assert self.cfg.dataset_type == 'graph_net'
        assert self.cfg.gn_classification == 'edge'

        # Load the network
        graph_net_cfg = copy.copy(self.cfg)
        graph_net_cfg.dataset_type = 'graph_net'
        graph_net_cfg.behaviornet_type = 'graph_net'
        graph_net_cfg.ckpt_path = '/data/sem-nav/experiments/v0.2/graph-net-3/models/iteration-015000.model'
        graph_net_cfg.n_frames_per_sample = 20
        self.net = get_net(graph_net_cfg.behaviornet_type, graph_net_cfg)

        self.evaluator = GraphNetEvaluator(cfg=graph_net_cfg)
        self.decode_type = self.evaluator.get_decode_type('single_frame')

    def reset_goal(self):
        super(GraphNetNavigationBrain, self).reset_goal()
        self.last_predicted_node_name = None
        self.last_predicted_behavior_id = None
        self.last_executed_valid_behavior_id = 'fd'
        self.localized = False  # Used for determining when to provide GT subgraph input

    def get_behavior(self):
        """Get the current behavior that should be executed.
        """
        if self.localized is False:
            print('Using GT node position for initial graph input.')

            # If this is the first iteration of the episode, use the ground truth initial position
            # for constructing the subgraph
            self.last_predicted_node_name = self.current_node
            self.last_predicted_behavior_id = self.get_gt_behavior()
            try:
                assert (self.last_predicted_node_name is not None
                        and self.last_predicted_behavior_id is not None)
            except AssertionError:
                return 'fd'

            is_new_episode = True

            self.localized = True
        else:
            is_new_episode = False

        # Construct the subgraph from the most recently reported position (node/edge)
        dummy_item = {'depth': self.cur_depth}
        cur_graph_net_input = GraphNetEvaluator.construct_graph_net_input(
            self.cfg,
            self.sem_graph,
            self.last_predicted_node_name,
            self.last_predicted_behavior_id,
            self.decode_type,
            dummy_item,
            )

        graph_net_input = {
            'depth': self.cur_depth,
            'graph_net_input': cur_graph_net_input,
            }
        output = self.evaluator.predict(self.net, graph_net_input, is_new_episode=is_new_episode)

        # Decode the output of GraphNetEvaluator.predict() and update the predicted location
        _, self.last_predicted_node_name, self.last_predicted_behavior_id = output

        # Once we have localized ourselves, figure out which behavior to execute
        edge = self.nav_plan.node2edge.get(self.sem_graph.nodes[self.last_predicted_node_name])
        if edge is not None:
            exec_behavior_id = behavior_id2category(edge.behavior_id).name
            self.last_executed_valid_behavior_id = exec_behavior_id
            return exec_behavior_id
        else:
            return self.last_executed_valid_behavior_id


if __name__ == '__main__':
    try:
        GraphNetNavigationBrain(mode='sim')
    except rospy.ROSInterruptException:
        rospy.loginfo('GraphNet Navigation Brain terminated.')
