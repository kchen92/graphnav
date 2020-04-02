from __future__ import print_function

import copy
import numpy as np
import rospy
import torch

import actionlib

from collections import deque
from enum import Enum, unique

from semnav.learning import get_net, merge_input_tensors
from semnav.learning.behavior_net.behavior_evaluator import BehaviorEvaluator
from semnav.lib.behavior_detectors import BehaviorDetector
from semnav.lib.behavior_executor import BehaviorExecutor
from semnav.lib.categories import SemanticCategory
from semnav.lib.image_map_transformer import Image2MapTransformer
from semnav.lib.navigation_plan import NavigationGoal, NavigationPlan
from semnav.lib.sem_graph import SemGraph
from semnav.lib.utils import compute_dist, compute_angle_delta

from semnav_ros.msg import NavCommandAction, NavCommandFeedback, NavCommandResult


@unique
class BrainState(Enum):
    FINISHED = 0
    EXECUTING = 1


@unique
class EpisodeResult(Enum):
    SUCCESS = 0
    DEVIATED = 1
    STUCK = 2
    ABORTED = 3


class NavigationBrain(BehaviorExecutor):
    """High-level navigation planner. Allows creation of navigation goals, determines the
    sequence of behaviors to be executed, and selects the execution behavior on-the-fly.
    """

    def __init__(self, node_name, mode, yaml_filepath, n_recent_poses=50, stuck_threshold_dist=0.5,
                 stuck_threshold_angle=20):
        """Constructor.

        Args:
            yaml_filepath: Path to area yaml.
            n_recent_poses: Integer used for determining when the robot is stuck. Compares the
                current agent position with the position n_recent_poses frames/updates ago.
            stuck_threshold_dist: Float (meters) used in conjunction with n_recent_poses. Checks
                whether the distance traveled between current frame and n_recent_poses
                frames/updates ago is within the stuck_threshold_dist.
        """
        assert mode == 'sim'  # Only sim is supported for now
        super(NavigationBrain, self).__init__(node_name, mode=mode)
        self.sem_graph = SemGraph(yaml_filepath)
        self.behavior_detector = BehaviorDetector()
        self.n_recent_poses = n_recent_poses
        self.stuck_threshold_dist = stuck_threshold_dist
        self.stuck_threshold_angle_radians = float(stuck_threshold_angle) / 180 * np.pi
        self.step_frequency = 5  # Hz
        self.rate = rospy.Rate(self.step_frequency)  # Hz
        self.feedback_frequency = 2  # Hz
        self.feedback_frequency_in_steps = self.step_frequency / self.feedback_frequency

        if mode == 'sim':
            self.img2map = Image2MapTransformer(yaml_filepath)

        self.load_behavior_nets()

        self.reset_goal()

        self.action_server = actionlib.SimpleActionServer('navigation_command', NavCommandAction,
                                                          execute_cb=self.action_server_cb,
                                                          auto_start=False)
        self.action_server.register_preempt_callback(self.action_server_preempt_cb)
        self.action_server.start()

    @property
    def recent_poses(self):
        return self._recent_poses

    @property
    def nav_goal(self):
        return self.nav_plan.nav_goal

    @property
    def nav_plan(self):
        return self._nav_plan

    @property
    def state(self):
        return self._state

    @property
    def current_node(self):
        return self._current_node

    def load_behavior_nets(self):
        # Turn left behavior
        tl_cfg = copy.copy(self.cfg)
        tl_cfg.dataset_type = 'temporal'
        tl_cfg.behaviornet_type = 'behavior_rnn'
        tl_cfg.ckpt_path = '/cvgl2/u/kchen92/sem-nav/experiments/v0.2/behavior_rnn-2-layer-lstm-512-hidden-size/tl/models/iteration-005000.model'
        tl_cfg.n_frames_per_sample = 20

        # Turn right behavior
        tr_cfg = copy.copy(self.cfg)
        tr_cfg.dataset_type = 'temporal'
        tr_cfg.behaviornet_type = 'behavior_rnn'
        tr_cfg.ckpt_path = '/cvgl2/u/kchen92/sem-nav/experiments/v0.2/behavior_rnn-2-layer-lstm-512-hidden-size/tr/models/iteration-005000.model'
        tr_cfg.n_frames_per_sample = 20

        # Go straight behavior
        s_cfg = copy.copy(self.cfg)
        s_cfg.dataset_type = 'temporal'
        s_cfg.behaviornet_type = 'behavior_rnn'
        s_cfg.ckpt_path = '/cvgl2/u/kchen92/sem-nav/experiments/v0.2/behavior_rnn-2-layer-lstm-512-hidden-size/s/models/iteration-007500.model'
        s_cfg.n_frames_per_sample = 20

        # Corridor follow behavior
        cf_cfg = copy.copy(self.cfg)
        cf_cfg.dataset_type = 'temporal'
        cf_cfg.behaviornet_type = 'behavior_cnn'
        cf_cfg.ckpt_path = '/cvgl2/u/kchen92/sem-nav/experiments/v0.2/behavior_cnn_stack-less-freq-val-2/cf/models/iteration-019000.model'
        cf_cfg.n_frames_per_sample = 20

        # Find door behavior
        fd_cfg = copy.copy(self.cfg)
        fd_cfg.dataset_type = 'temporal'
        fd_cfg.behaviornet_type = 'behavior_cnn'
        fd_cfg.ckpt_path = '/cvgl2/u/kchen92/sem-nav/experiments/v0.2/behavior_cnn_stack-less-freq-val-2/fd/models/iteration-017000.model'
        fd_cfg.n_frames_per_sample = 20

        behavior_cfgs = {
            'tl': tl_cfg,
            'tr': tr_cfg,
            's': s_cfg,
            'cf': cf_cfg,
            'fd': fd_cfg,
            }

        # Assume cfg.n_frames_per_sample is the same for all networks
        all_n_frames_per_sample = []
        for cur_cfg in behavior_cfgs.values():
            if cur_cfg.n_frames_per_sample is not None:
                all_n_frames_per_sample.append(cur_cfg.n_frames_per_sample)
        if len(all_n_frames_per_sample) > 0:
            assert len(set(all_n_frames_per_sample)) == 1
            self.n_frames_per_sample = all_n_frames_per_sample[0]
        else:
            self.n_frames_per_sample = None

        # Load networks
        print('Loading networks...')
        self.behavior_nets = {behavior_id: get_net(behavior_cfg.behaviornet_type, behavior_cfg)
                              for behavior_id, behavior_cfg in behavior_cfgs.iteritems()}
        print('Mapping fd_r, fd_l to fd network')
        self.behavior_nets['fd_r'] = self.behavior_nets['fd']
        self.behavior_nets['fd_l'] = self.behavior_nets['fd']
        print('Mapping s_r, s_l to s network')
        self.behavior_nets['s_r'] = self.behavior_nets['s']
        self.behavior_nets['s_l'] = self.behavior_nets['s']
        print('Networks loaded')

        # Set all networks to eval mode
        for cur_net in self.behavior_nets.values():
            cur_net.eval()

        # Create evaluators
        self.evaluators = {behavior_id: BehaviorEvaluator(cfg=behavior_cfg)
                           for behavior_id, behavior_cfg in behavior_cfgs.iteritems()}
        print('Mapping fd_r, fd_l to fd network')
        self.evaluators['fd_r'] = self.evaluators['fd']
        self.evaluators['fd_l'] = self.evaluators['fd']
        print('Mapping s_r, s_l to s network')
        self.evaluators['s_r'] = self.evaluators['s']
        self.evaluators['s_l'] = self.evaluators['s']

        self.behavior_cfgs = behavior_cfgs

    def run_episode(self):
        """Run the main loop for the navigation brain.
        """
        steps_since_last_feedback_publish = 0
        with torch.no_grad():
            while not self.is_episode_finished():
                self.update_brain()
                behavior_id = self.get_behavior()
                self.execute_behavior(behavior_id)

                # Publish feedback
                steps_since_last_feedback_publish += 1
                if steps_since_last_feedback_publish > self.feedback_frequency_in_steps:
                    self.action_server_publish_feedback(behavior_id)
                    steps_since_last_feedback_publish = 0

                self.rate.sleep()
        if self.is_episode_successful() is True:
            result = EpisodeResult.SUCCESS
        elif self.is_deviated() is True:
            result = EpisodeResult.DEVIATED
        elif self.is_stuck() is True:
            result = EpisodeResult.STUCK
        else:
            raise ValueError
        print(result)
        return result

    def update_visuals(self):
        """Update any visuals used by the brain/behavior networks. This is called from
        self.update_brain().
        """
        # Read depth image
        cur_depth = self.depth_transform(self.last_depth)  # Shape (H, W)
        cur_depth = torch.from_numpy(cur_depth[np.newaxis, np.newaxis, :]).to(self.cfg.device)

        # Create a new image stack if at beginning of sim
        if (self.n_frames_per_sample is not None) and (self.depth_stack is None):
            self.depth_stack = cur_depth.repeat(1, self.n_frames_per_sample, 1, 1)

        # Keep an updated image stack if necessary
        if (self.n_frames_per_sample is not None) and (self.depth_stack is not None):
            # Most recent image is at the end
            self.depth_stack = torch.cat([self.depth_stack[:, 1:, :, :], cur_depth], dim=1)

        self.cur_depth = cur_depth

    def update_sim_info(self):
        """Update brain with simulation information.
        """
        # Update the location in the topological map
        room_name = self.img2map.get_room_name(self.last_position)
        if room_name is None:
            print('Lost! Could not find current room name.')
            self._current_node = None
        else:
            self._current_node = self.sem_graph.get_closest_node(room_name, self.last_position,
                                                                 self.last_orientation,
                                                                 alignment_matters=True)

            # Make sure current node is correctly set when in a room, even when agent does not face
            # the correct direction (similar to if alignment_matters is False)
            if (self.behavior_detector.in_room_by_name(room_name)
                    and (self.last_valid_node is not None)
                    and ((self.current_node is None)
                         or (not self.behavior_detector.in_room_by_name(self.current_node.name)))):
                if room_name in self.sem_graph.nodes:
                    self._current_node = self.sem_graph.nodes[room_name]

            if self.current_node in self.nav_plan.node_list:
                self.last_valid_node = self.current_node

        # Update self.recent_poses
        last_pose = (self.last_position, self.last_orientation)
        if len(self.recent_poses) < self.n_recent_poses:
            self._recent_poses.append(last_pose)
        else:  # queue is full
            self._recent_poses.popleft()
            self._recent_poses.append(last_pose)

    def update_brain(self):
        """Update the brain with the latest information, such as visual input and location in map.
        Keeping the brain updated allows it to also update/localize the agent in the topological map.
        """
        if self.mode == 'sim':
            self.update_sim_info()
        self.update_visuals()

    def get_gt_behavior(self):
        """Using the brain's (GT) estimate of the agent's position in the topological map, return
        the correct behavior for getting to the current navigation goal.

        Returns:
            behavior_id: The correct behavior_id to execute if at a valid node on the current
                navigation plan. Otherwise, return None.
        """
        if self.nav_plan is None:
            return None

        edge = self.nav_plan.node2edge.get(self.current_node)
        if edge is not None:
            return edge.behavior_id
        else:
            if self.current_node is not None:
                print('Current node not part of navigation plan:', self.current_node.name)
            return None

    def execute_behavior(self, behavior_id):
        """Execute the behavior specified by behavior_id.

        Args:
            behavior_id: A behavior ID. If this is None, the behavior ID is assumed to be STOP.
        """
        if behavior_id is None:  # Nonbehavior keypress
            exec_behavior_id = 'STOP'
        else:  # If valid behavior keypress
            exec_behavior_id = behavior_id  # Execute new behavior

        # Check if we are executing a new/different behavior compared to previous exec behavior
        if exec_behavior_id != self.exec_behavior_id:
            is_exec_new_behavior_id = True
        else:
            is_exec_new_behavior_id = False

        self.exec_behavior_id = exec_behavior_id

        if exec_behavior_id == 'STOP':
            output_vel = torch.zeros(1, 2)
        else:
            net = self.behavior_nets[exec_behavior_id]
            evaluator = self.evaluators[exec_behavior_id]

            cur_behavior_cfg = self.behavior_cfgs[exec_behavior_id]
            if cur_behavior_cfg.use_semantic_class is None:  # Depth only input
                # When we execute a new behavior, reset the internal depth stack
                if (self.n_frames_per_sample is not None) and (is_exec_new_behavior_id is True):
                    evaluator.depth_stack = self.depth_stack

                cur_input = self.cur_depth
            else:  # Depth + semantics input
                semantic_mask_stack = self.semantic_stack == SemanticCategory[cur_behavior_cfg.use_semantic_class]
                semantic_mask_stack = semantic_mask_stack.type(torch.FloatTensor).to(self.cfg.device)
                depth_semantic_stack = merge_input_tensors([self.depth_stack, semantic_mask_stack])

                # When we execute a new behavior, reset the internal depth/semantic stack
                if (self.n_frames_per_sample is not None) and (behavior_id is not None):
                    evaluator.depth_stack = depth_semantic_stack

                cur_semantic = self.cur_semantic == SemanticCategory[cur_behavior_cfg.use_semantic_class]
                cur_semantic = cur_semantic.type(torch.FloatTensor).to(self.cfg.device)
                cur_frame = torch.cat([self.cur_depth, cur_semantic], dim=1)

                cur_input = cur_frame
            if net.is_recurrent is True:
                # For recurrent models, we want to set is_new_episode to True whenever a behavior
                # switch occurs so that the hidden state can reset
                if is_exec_new_behavior_id is True:
                    is_new_episode = True
                else:
                    is_new_episode = False
            else:
                # For simple feedforward networks, we always set is_new_episode to False because
                # we already manually updated the evaluator depth stack
                is_new_episode = False
            output_vel = evaluator.predict(net, cur_input, is_new_episode=is_new_episode)
        self.execute_vel(output_vel)

    def is_stuck(self):
        """Return whether the agent is stuck or not.
        """
        assert self.nav_plan is not None
        if len(self.recent_poses) < self.n_recent_poses:
            return False
        position_delta = compute_dist(self.recent_poses[0][0][:2],
                                      self.recent_poses[-1][0][:2])
        orientation_delta = compute_angle_delta(self.recent_poses[0][1][2],
                                                self.recent_poses[-1][1][2])
        if ((position_delta < self.stuck_threshold_dist)
                and (orientation_delta < self.stuck_threshold_angle_radians)):
            return True
        else:
            return False

    def is_deviated(self):
        """Whether the agent has deviated from the navigation goal (visited a node not in the
        navigation plan).
        """
        assert self.nav_plan is not None
        return (self.current_node is not None) and (self.current_node not in self.nav_plan.node_list)

    def _is_at_nav_goal(self):
        """Whether the robot is currently at the navigation goal or not.
        """
        return self.current_node is self.nav_goal.end_node

    def is_episode_finished(self):
        """Report whether agent is done with attempting the current navigation goal. This can be due
        to being stuck, traveling to the wrong location, or successfully finishing the navigation goal.
        """
        assert self.nav_plan is not None
        if self.is_stuck() or self.is_deviated() or self._is_at_nav_goal():
            self._state = BrainState.FINISHED
        return self.state is BrainState.FINISHED

    def is_episode_successful(self):
        """Check if the episode is successful. This should only be called after
        self.is_episode_finished() returns True. Otherwise, this method will return None.
        """
        assert self.nav_plan is not None
        if self.state is not BrainState.FINISHED:
            return None
        return self._is_at_nav_goal()

    def reset_goal(self):
        """Clear/forget the current goal (so that there is no goal).
        """
        self._nav_plan = None
        self._state = BrainState.FINISHED
        self._recent_poses = deque([], maxlen=self.n_recent_poses)
        self._current_node = None
        self.last_valid_node = None
        self.depth_stack = None
        self.semantic_stack = None
        self.exec_behavior_id = None

    def execute_nav_plan(self, nav_plan):
        """Set a new navigation goal, overwriting the current one if it exists.

        Args:
            start_node: Start node.
            end_node: End/destination node.
            shortest_path: Boolean for whether to use the shortest path to destination or not.
        """
        if self.state is not BrainState.FINISHED:
            print('Failed to set goal. Brain is still executing a previous navigation goal.')
            result = EpisodeResult.ABORTED
            return result

        self.reset_goal()
        self._state = BrainState.EXECUTING
        self._nav_plan = nav_plan
        result = self.run_episode()
        print(result)
        return result

    def execute_nav_goal_by_name(self, start_node_name, end_node_name):
        """Create and execute a shortest path NavigationPlan from start_node_name to end_node_name.
        This method of commanding the NavigationBrain bypasses the action server. By default, a
        NavigationPlanner should be used for sending commands to the NavigationBrain.
        """
        start_node = self.sem_graph.nodes[start_node_name]
        end_node = self.sem_graph.nodes[end_node_name]
        nav_goal = NavigationGoal(start_node, end_node)
        node_list = self.sem_graph.find_shortest_path(nav_goal.start_node,
                                                      nav_goal.end_node)
        nav_plan = NavigationPlan(node_list=node_list)
        return self.execute_nav_plan(nav_plan)

    def action_server_cb(self, nav_plan_msg):
        episode_idx, nav_plan = NavigationPlan.from_msg(self.sem_graph, nav_plan_msg)
        print('Received goal: {} to {}'.format(nav_plan.nav_goal.start_node.name,
                                               nav_plan.nav_goal.end_node.name))
        result = self.execute_nav_plan(nav_plan)
        rospy.loginfo('Episode {}: {}'.format(episode_idx, result))
        if self.last_valid_node is None:
            percentage_plan_completed = 0.
        else:
            percentage_plan_completed = self.nav_plan.percentage_plan_completed(self.last_valid_node)
        result_msg = NavCommandResult(episode_result=result.value,
                                      percentage_plan_completed=percentage_plan_completed)
        if result is EpisodeResult.SUCCESS:
            self.action_server.set_succeeded(result_msg)
        else:
            self.action_server.set_aborted(result_msg)

    def action_server_preempt_cb(self):
        rospy.loginfo('Preempted!')
        self.action_server.set_preempted()

    def action_server_publish_feedback(self, behavior_id):
        feedback = NavCommandFeedback()

        # Current node
        if self.current_node is None:
            feedback.cur_node_name = ''
        else:
            feedback.cur_node_name = self.current_node.name
            print('Current node: {}'.format(self.current_node.name))

        # Current room
        cur_room_name = self.img2map.get_room_name(self.last_position)
        if cur_room_name is None:
            feedback.cur_room_name = ''
        else:
            feedback.cur_room_name = cur_room_name

        # Current GT behavior
        gt_behavior = self.get_gt_behavior()
        if gt_behavior is None:
            feedback.gt_behavior_id = ''
        else:
            feedback.gt_behavior_id = gt_behavior

        # Current executing behavior
        if behavior_id is None:
            feedback.exec_behavior_id = ''
        else:
            feedback.exec_behavior_id = behavior_id
            print('Executing Behavior: {}'.format(self.exec_behavior_id))
        print('-------------')

        self.action_server.publish_feedback(feedback)
