"""Robot navigator.

Based on Patrick's RobotNavigator.
"""

import rospy
import actionlib

from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionResult
from nav_msgs.srv import GetPlan, GetPlanRequest
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from rosgraph_msgs.msg import Log
from std_msgs.msg import Bool
from tf.transformations import quaternion_from_euler
from math import pow, sqrt


class RobotNavigator(object):

    def __init__(self, move_base_timeout=60, pause_interval=3, map_frame='map'):
        # Maximum time to wait (in seconds) for a goal to be reached
        self.move_base_timeout = move_base_timeout

        # How long to wait in between goals
        self.pause_interval = pause_interval
        self.map_frame = map_frame

        # Goal state return values
        self.goal_states = ['PENDING', 'ACTIVE', 'PREEMPTED',
                            'SUCCEEDED', 'ABORTED', 'REJECTED',
                            'PREEMPTING', 'RECALLING', 'RECALLED',
                            'LOST']

        # Keep track of the robot's position as returned by AMCL or Ground Truth
        self.robot_pose = PoseStamped()

        # Publisher to manually control the robot (e.g. to stop it, queue_size=5)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=5)

        # Publisher for manually operated goals
        self.goal_pose_pub = rospy.Publisher('move_base/current_goal', PoseStamped, queue_size=5)

        # Publish results for manually operated goals
        self.goal_result_pub = rospy.Publisher('move_base/result', MoveBaseActionResult, queue_size=5)

        # Subscribe to the move_base action server
        self.move_base = actionlib.SimpleActionClient('move_base', MoveBaseAction)

        rospy.loginfo('Waiting for move_base action server...')

        # Wait up to 10 seconds for the action server to become available
        self.move_base.wait_for_server(rospy.Duration(10.0))

        rospy.loginfo('Connected to move base server')

        # Connect to the move_base/clear_costmaps service
        self.clear_costmaps = rospy.ServiceProxy('move_base/clear_costmaps', Empty)

        # A service to get a global plan
        self.get_plan_service = rospy.ServiceProxy('move_base/make_plan', GetPlan, persistent=True)

        # Publisher to clear the current goal

        # Subscribe to ROS log messages so we can see when move_base is in trouble
        rospy.Subscriber('rosout_agg', Log, self.get_log_messages)

        # Cancel any left over goals
        self.cancel_nav_goals()

        # A variable to hold the initial pose of the robot to be set by the user in RViz
        self.initial_pose = None

        # Variables to keep track of success rate, running time, and distance traveled
        self.n_runs = 0
        self.n_successes = 0
        self.distance_traveled = 0
        self.start_time = rospy.Time.now()
        self.running_time = 0

        rospy.Subscriber('initialpose', PoseWithCovarianceStamped, self.update_initial_pose)

        self.amcl_sub = rospy.Subscriber('amcl_pose_ground_truth', PoseWithCovarianceStamped, self.amcl_callback)
        rospy.loginfo("Waiting for robot pose...")  # rospy.loginfo("Waiting for AMCL...")
        rospy.wait_for_message('amcl_pose_ground_truth', PoseWithCovarianceStamped)

        # TODO: What does this do?
        while self.robot_pose == PoseStamped():
            rospy.sleep(0.1)

        # We need this variable to track distance travelled
        self.last_pose = self.robot_pose

        rospy.loginfo('Robot Navigator ready.')

    def cancel_nav_goals(self):
        """
        Cancel navigation goals
        """
        self.move_base.cancel_all_goals()
        # Wait a moment for them to cancel
        rospy.sleep(1.0)

    def set_goal(self, x, y, z, theta, goal_frame='map'):
        # Execute move_base normally
        robot_stuck = self.send_goal(x, y, z, theta)

        rospy.loginfo('Pausing for ' + str(self.pause_interval) + ' seconds...')
        rospy.sleep(self.pause_interval)

        return robot_stuck

    def send_goal(self, x, y, z, theta, goal_frame='map'):
        # Set up the next goal location
        goal = MoveBaseGoal()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y

        quat = quaternion_from_euler(0, 0, theta)

        goal.target_pose.pose.orientation.x = quat[0]
        goal.target_pose.pose.orientation.y = quat[1]
        goal.target_pose.pose.orientation.z = quat[2]
        goal.target_pose.pose.orientation.w = quat[3]

        goal.target_pose.header.frame_id = goal_frame
        goal.target_pose.header.stamp = rospy.Time.now()

        # Let the user know where the robot is going next
        rospy.loginfo('Going to: (' + str(x) + ', ' + str(y) + ')')

        # Increment the counter
        self.n_runs += 1

        # Start the robot toward the next location
        self.move_base.send_goal(goal, done_cb=self.nav_done_cb, feedback_cb=self.nav_feedback_cb, active_cb=self.nav_active_cb)

        rospy.loginfo('Waiting for result...')

        # This flag is set to True in the nav_done_cb() callback
        self.nav_action_finished = False

        # Flag to indicate if the robot got stuck
        self.robot_stuck = False

        # A timer to test for a timeout
        start_time = rospy.Time.now()

        state = None

        # We cannot use the move_base.wait_for_result() method here as it will block the entire
        # script so we break it down into small time slices
        while not self.nav_action_finished and not self.robot_stuck:
            if (rospy.Time.now() - start_time).to_sec() > self.move_base_timeout:
                self.move_base.cancel_goal()
                rospy.loginfo('Timed out achieving goal')
                state = GoalStatus.ABORTED
                break

            rospy.sleep(0.05)

        if state is None:
            state = self.move_base.get_state()

        if state == GoalStatus.SUCCEEDED:
            rospy.loginfo('Goal succeeded!')
            self.n_successes += 1
        else:
            rospy.loginfo('Goal failed with error code: ' + str(self.goal_states[state]))
            if state == GoalStatus.ABORTED:
                self.robot_stuck = True

        # How long have we been running?
        running_time = rospy.Time.now() - self.start_time
        running_time = running_time.secs / 60.0

        # Print a summary success/failure, distance traveled and time elapsed
        try:
            rospy.loginfo('Success so far: ' + str(self.n_successes) + '/' +
                          str(self.n_runs) + ' = ' +
                          str(100 * self.n_successes / self.n_runs) + '%')
            rospy.loginfo('Running time: ' + str(trunc(running_time, 1)) +
                          ' min Distance: ' + str(trunc(self.distance_traveled, 1)) +
                          ' m Ave Speed: ' + str(trunc(self.distance_traveled / running_time / 60.0, 2)) + ' m/s')
        except ZeroDivisionError:
            # Division by zero if robot is already at first goal location
            pass

        return self.robot_stuck

    def amcl_callback(self, msg):
        self.robot_pose = msg.pose

    def nav_active_cb(self):
        rospy.loginfo('Goal activated.')

    def nav_done_cb(self, status, result):
        if status == GoalStatus.PREEMPTED:
            rospy.loginfo('PREEMPTED!')

        self.clear_costmaps()
        self.nav_action_finished = True

    def nav_feedback_cb(self, msg):
        distance = sqrt(pow(self.robot_pose.pose.position.x - self.last_pose.pose.position.x, 2) +
                        pow(self.robot_pose.pose.position.y - self.last_pose.pose.position.y, 2))

        self.last_pose.pose = msg.base_position.pose

        self.distance_traveled += distance  # TODO: Why do we care about total distance traveled among all episodes?

    def update_initial_pose(self, initial_pose):
        self.initial_pose = initial_pose

    def get_log_messages(self, msg):
        stuck_msgs = ('Aborting', 'Clearing costmap', 'recovery behaviors')
        failure_msgs = ('Aborted', 'ABORTED')
        if msg.name == '/move_base':
            # rospy.loginfo(msg.msg)  # Gibson node is already printing this

            if any(substring in msg.msg for substring in stuck_msgs):
                rospy.logwarn(msg.msg)
                rospy.logwarn('move_base says robot is stuck!')
                self.robot_stuck = True
                self.nav_action_finished = True

            elif any(substring in msg.msg for substring in failure_msgs):
                rospy.logwarn(msg.msg)
                rospy.logwarn('Goal not reachable. Getting new goal.')
                self.clear_costmaps()
                self.nav_action_finished = True

    def shutdown(self):
        rospy.loginfo('Stopping the robot...')
        self.cancel_nav_goals()
        rospy.sleep(2)
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)


def trunc(f, n):
    # Truncates/pads a float f to n decimal places without rounding
    slen = len('%.*f' % (n, f))
    return float(str(f)[:slen])
