from copy import deepcopy
import warnings

import gym

from envs.wrappers.multitask import MultitaskWrapper
from envs.wrappers.pixels import PixelWrapper
from envs.wrappers.tensor import TensorWrapper


from envs.gazebo import make_env as make_gazebo_env

warnings.filterwarnings('ignore', category=DeprecationWarning)

def make_env(cfg):
	"""
	Make an environment for TD-MPC2 experiments.
	"""
	gym.logger.set_level(40)

	env = make_gazebo_env(cfg)

	if env is None:
		raise ValueError(f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.')		
	env = TensorWrapper(env)
	if cfg.get('obs', 'state') == 'rgb':
		env = PixelWrapper(cfg, env)
	try: # Dict
		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
	except: # Box
		cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
	cfg.action_dim = env.action_space.shape[0]
	cfg.episode_length = env.max_episode_steps
	cfg.seed_steps = max(1000, 5*cfg.episode_length)
	return env
