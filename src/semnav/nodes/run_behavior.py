#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import torch
import rospy

from semnav.learning import get_net
from semnav.learning.behavior_net.behavior_evaluator import BehaviorEvaluator
from semnav.lib.behavior_executor import BehaviorExecutor


class BehaviorNode(BehaviorExecutor):
    """Execute a behavior using a behavior network.
    """

    def __init__(self, mode):
        """Constructor.

        Args:
            mode: 'real' or 'sim', it will change which topics to subscribe.
        """
        super(BehaviorNode, self).__init__(node_name='behavior_node', mode=mode)
        print('Loading network...')
        self.net = get_net(self.cfg.behaviornet_type, self.cfg)
        self.net.eval()  # Set to eval mode
        print('Network loaded')
        self.evaluator = BehaviorEvaluator()

        while self.last_depth is None:
            rospy.sleep(0.2)  # Let some messages come in

        rate = rospy.Rate(5)  # 5 Hz
        is_new_episode = True
        print('Running loop...')
        with torch.no_grad():
            while not rospy.is_shutdown():
                cur_depth = self.depth_transform(self.last_depth)  # Shape (H, W)
                # Add batch and channel dimensions
                cur_depth = torch.from_numpy(cur_depth[np.newaxis, np.newaxis, :]).to(self.cfg.device)
                output_vel = self.evaluator.predict(self.net, cur_depth, is_new_episode)
                self.execute_vel(output_vel)
                is_new_episode = False
                rate.sleep()


if __name__ == '__main__':
    try:
        BehaviorNode(mode='sim')
    except rospy.ROSInterruptException:
        rospy.loginfo('Behavior node terminated.')
