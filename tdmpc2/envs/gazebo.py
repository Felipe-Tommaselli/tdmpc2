import os
import subprocess
import time
from os import path
import numpy as np
import rospy
import colorful as cf 
import torch

import gym

from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from std_msgs.msg import Float32MultiArray
from scipy.special import softmax
TIME_DELTA = 0.1

from collections import defaultdict # for info data

def normalize(input_image):
    #input is c,h,w which we change to w,h,c
    input_image = input_image.transpose((1,2,0))

    output = input_image.copy()

    r = input_image[:,:,0]
    g = input_image[:,:,1]
    b = input_image[:,:,2]

    r_min = r.min()
    g_min = g.min()
    b_min = b.min()

    r_max = r.max()
    g_max = g.max()
    b_max = b.max()

    output[:,:,0] = (input_image[:,:,2]-b_min)/(b_max-b_min)
    output[:,:,1] = (input_image[:,:,1]-g_min)/(g_max-g_min)
    output[:,:,2] = (input_image[:,:,0]-r_min)/(r_max-r_min)

    output = np.where(output > 0.1, output, np.zeros_like(output))

    #TODO: check if this is necessary
    output = output.transpose((2, 0, 1))

    return output


class GazeboEnv(gym.Env):
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, cfg):
        super(GazeboEnv, self).__init__()
        
        self.cfg = cfg
        self.obs_dim = cfg.get("obs_shape")
        self.action_dim = cfg.get("action_dim")
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        #self.reward_range = (-np.inf, np.inf)
        #self.metadata = {'render.modes': ['human']}
        #self.spec = None


        self.odom_x = 0
        self.odom_y = 0
        self.vel_x = 1
        self.pitch = 0
        self.roll = 0

        self.goal_x = 1
        self.goal_y = 0.0

        self.upper = 5.0
        self.lower = -5.0
        self.last_odom = Odometry()
        self.ll_odom_x = 0.0
        self.delta_x = 0.0

        self.last_heat_map = np.ones((3, 56, 80))
        self.state = np.zeros((3, 56, 80))
        self.reward = 0
        self.done = False 
        self.info = defaultdict(float, {'success': 0.0})
        
        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        self.dis_error=0

        port = "11311"
        subprocess.Popen(["roscore", "-p", port])

        print("Roscore launched!")

        # Launch the simulation with the given launchfile name
        rospy.init_node("gym", anonymous=True)
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "launchs", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched!")

        # Set up the ROS publishers and subscribers
        # self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.terra_vel_pub = rospy.Publisher("/terrasentia/cmd_vel", TwistStamped, queue_size=10)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)

        self.odom = rospy.Subscriber(
            "/terrasentia/ground_truth", Odometry, self.odom_callback, queue_size=1
        )
        
        self.heat_map=rospy.Subscriber(
            "/terrasentia/vision/keypoint_heatmap", Float32MultiArray, self.heat_map_callback, queue_size=1
        )

        self.keypoints = [0]*6
        self.keypoints_sub = rospy.Subscriber(
            "/terrasentia/vision/keypoint", Float32MultiArray, self.keypoints_callback, queue_size=1
        )

        self.heading_error = rospy.Subscriber(
            "/terrasentia/heading_error", Float32MultiArray, self.heading_error_callback, queue_size=1
        )
        self.d_error=rospy.Subscriber(
            "/terrasentia/distance_error", Float32MultiArray, self.d_error_callback, queue_size=1
        )

    def keypoints_callback(self, keypoints_data):
        keypoints = keypoints_data.data
        self.keypoints = np.array([x / 80 if i % 2 == 0 else x / 56 for i, x in enumerate(keypoints)])

    def heading_error_callback(self,head_erro_data):
        self.heading_error=float(head_erro_data.data[0])
    
    def heat_map_callback(self, heat_map_data):
        heat_map_data=np.array(heat_map_data.data)
        data = heat_map_data.reshape(3,56,80)
        data[0] = softmax(data[0])
        data[1] = softmax(data[1])
        data[2] = softmax(data[2])
        data = normalize(data)
        self.last_heat_map = data

    def d_error_callback(self, dis_error):
        self.dis_error=float(dis_error.data[0])
        
    def odom_callback(self, od_data):
        self.last_odom = od_data
        self.vel_x = od_data.twist.twist.linear.x
        self.pitch = od_data.twist.twist.angular.y 
        self.roll  = od_data.twist.twist.angular.x

    # Perform an action and read a new state
    def step(self, action):
        #TODO: get the current action pair format from the agent
        
        target = False

        # Publish the robot action
        vel_cmd = TwistStamped()
        vel_cmd.twist.linear.x = action[0]
        vel_cmd.twist.angular.z = action[1]
        self.terra_vel_pub.publish(vel_cmd)
        # self.publish_markers(action)

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # # Calculate robot heading from odometry data
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        int_odom_x = int(self.odom_x*10)
        self.delta_x = int_odom_x - self.ll_odom_x*10
        self.ll_odom_x = int_odom_x/10


        #robot_state = [action[0], action[1]]
        vision_state = [self.last_heat_map[:]]
        #self.state = np.append(vision_state, robot_state)

        if isinstance(vision_state, int):
            self.state = np.array(vision_state[0])
        else:
            self.state = np.array(vision_state)
        
        collision = self.observe_collision(self.dis_error, self.vel_x, vel_cmd.twist.linear.x, self.pitch, self.roll)
        self.reward = self.get_reward(self.dis_error, self.delta_x, collision, action)
        self.done = collision

        if self.done == True:
            self.ll_odom_x = 0

        #TODO: change info value when necessary 
        #self.info['success'] = 1.0 if self.odom_x > 1.0 else 0.0

        #TODO: check obs values min and max

        #* -------- ENVIROMENT -------- 
        obs = torch.tensor(self.state.flatten())
        reward = self.reward if isinstance(self.reward, torch.Tensor) else torch.tensor(self.reward)
        done = self.done
        info = self.info

        return obs, reward, done, info


    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")


        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        
        #robot_state = [0.0, 0.0]
        vision_state = [self.last_heat_map[:]]
        #self.state = np.append(vision_state, robot_state)
        if isinstance(vision_state, int):
            self.state = np.array(vision_state[0])
        else:
            self.state = np.array(vision_state)

        #* -------- ENVIROMENT -------- 
        obs = torch.tensor(self.state.flatten())
        
        return obs

    @staticmethod
    def observe_collision(distance_error, vel_x, vel_cmd, pitch, roll):
        if abs(vel_x) < 0.15 and abs(vel_cmd) > 0.25:
            return True
        if abs(distance_error) > 0.2:
            return True
        elif abs(pitch) > 0.01 or abs(roll) > 0.01:
            return True
        return False

    @staticmethod
    def get_reward(distance_error, delta_x, collision, action):
        # if delta_x <0:
        #     delta_x=0
        if collision:
            # self.ll_odom_x = 0.0
            return -100.0
        else:
            return action[0]/2 - abs(action[1]) + delta_x/10

##############################################

def make_env(cfg):
    print('MAKE GAZEBO ENV')
    env = GazeboEnv("multi_robot_scenario.launch", cfg)
    # env = ActionDTypeWrapper(env, np.float32)
    # env = ActionRepeatWrapper(env, 2)
    # env = action_scale.Wrapper(env, minimum=-1., maximum=1.)
    # env = ExtendedTimeStepWrapper(env)
    # env = TimeStepToGymWrapper(env, domain, task)
    return env
