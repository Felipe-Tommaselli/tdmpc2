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
from std_msgs.msg import String
from scipy.special import softmax

from sensor_msgs.msg import LaserScan

RESET_DELAY = 0.33
STEP_DELAY = 0.33

from collections import defaultdict # for info data

def normalize_image(input_image):
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

def normalize_actions(action):
    min_linear = 0.1 
    max_linear = 1.0
    min_angular = -0.9
    max_angular = 0.9

    # norm (-1,1) to (0,1)
    a_norm_linear = ((action[0] + 1)/2)
    a_norm_angular = ((action[1] + 1)/2)

    action[0] = a_norm_linear * (max_linear - min_linear) + min_linear
    action[1] = a_norm_angular * (max_angular - min_angular) + min_angular    

    return action

class GazeboEnv(gym.Env):
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, cfg):
        super(GazeboEnv, self).__init__()
        
        self.cfg = cfg
        self.obs_dim = cfg.get("obs_shape")
        self.action_dim = cfg.get("action_dim")
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
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
        #self.state = np.zeros((3, 56, 80))
        self.state = np.zeros(1081)
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
        self.terra_vel_pub = rospy.Publisher("/terrasentia/cmd_vel", TwistStamped, queue_size=10)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.collision_pub = rospy.Publisher("terrasentia/collision", String, queue_size=1)
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

        #TODO: debug scan error:
        # [ERROR] [1723491838.643326690, 0.001000000]: Client [/gym_273175_1723491816928] wants topic /terrasentia/scan to have datatype/md5sum [std_msgs/Float32MultiArray/6a40e0ffa6a17a503ac3f8616991b1f6], but our version has [sensor_msgs/LaserScan/90c7ef2dc6895d81024acba2ac42f369]. Dropping connection.

        self.scan = np.zeros(1081)
        self.scan_sub = rospy.Subscriber(
            "/terrasentia/scan", LaserScan, self.scan_callback, queue_size=2**28
        )

    
    def scan_callback(self, data):
        data_processed = [10.0 if (value == 'inf' or value == 'infinity') else value for value in data.ranges]
        self.scan = np.array(data_processed)  
        print(f'scan callback: {self.scan}')      

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
        data = normalize_image(data)
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
        target = False

        # normalize -1 <-> 1 to each action space
        action = normalize_actions(action)

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

        # wait for the robot to be in the position
        time.sleep(STEP_DELAY)

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


        ##robot_state = [action[0], action[1]] #TODO: remember to unclip 
        #vision_state = [self.last_heat_map[:]]
        ## self.state = np.append(vision_state, robot_state)


        # if isinstance(vision_state, int):
        #     self.state = np.array(vision_state[0])
        # else:
        #     self.state = np.array(vision_state)
        
        self.state = self.scan

        print(f'step state: {self.state}')

        collision = self.observe_collision(self.dis_error, self.vel_x, vel_cmd.twist.linear.x, self.pitch, self.roll)
        self.reward = self.get_reward(self.dis_error, self.delta_x, collision['response'], action)
        self.done = collision['response']

        if collision['response'] != False:
            self.collision_pub.publish(collision['type'])
            vel_cmd.twist.linear.x = 0
            vel_cmd.twist.angular.z = 0
            self.terra_vel_pub.publish(vel_cmd)
        
        if self.done == True:
            self.ll_odom_x = 0

        #TODO: change info value when necessary 
        #self.info['success'] = 1.0 if self.odom_x > 1.0 else 0.0

        #* -------- ENVIROMENT -------- 
        obs = torch.tensor(self.state.flatten())
        reward = self.reward if isinstance(self.reward, torch.Tensor) else torch.tensor(self.reward)
        done = self.done
        info = self.info

        print(f'step obs: {obs}')

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

        time.sleep(RESET_DELAY)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        
        # #robot_state = [0.0, 0.0]
        # vision_state = [self.last_heat_map[:]]
        # #self.state = np.append(vision_state, robot_state)
        # if isinstance(vision_state, int):
        #     self.state = np.array(vision_state[0])
        # else:
        #     self.state = np.array(vision_state)

        self.state = self.scan

        print(f'reset state: {self.state}')

        #* -------- ENVIROMENT -------- 
        obs = torch.tensor(self.state.flatten())
        
        print(f'reset obs: {obs}')
        return obs

    @staticmethod
    def observe_collision(distance_error, vel_x, vel_cmd, pitch, roll):
        if abs(vel_x) < 0.15 and abs(vel_cmd) > 0.25:
            return {'response':True, 'type': 'stuck'}
        
        if abs(distance_error) > 0.5:
            if distance_error < 0:
                return {'response':True, 'type': 'distance left'}
            else:
                return {'response':True, 'type': 'distance right'}
        
        elif abs(pitch) > 0.01 or abs(roll) > 0.01:
            return {'response':True, 'type': 'acrobatic'}
        
        return {'response':False, 'type': None}

    @staticmethod
    def get_reward(distance_error, delta_x, collision, action):
        if collision:
            pass
            #return -100.0
        else:
            pass 
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
