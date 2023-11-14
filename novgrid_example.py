import gym
import gym_minigrid
import novgrid
from gym import ObservationWrapper
from novgrid.novelty_generation.novelty_wrappers import *

import dreamerv2.api as dv2

        
config = dv2.defaults.update({
    #'logdir': '~/code/worldcloner_weights/dreamer_lavasafe_pretrain',
    'logdir': '~/logdir/dreamer_novtest',
    'log_every': 1e3,
    'train_every': 10,
    #'prefill': 1,
    'prefill': 1e5,
    'actor_ent': 3e-3,
    'loss_scales.kl': 1.0,
    'discount': 0.99,
}).parse_flags()

#env = gym.make('MiniGrid-LavaShortcutMaze8x8-v0')
#env = gym.make('MiniGrid-LavaSafeMaze8x8-v0')
env = gym.make('MiniGrid-DoorKey-8x8-v0')
env = DoorKeyChange(env,novelty_episode=1)
env = gym_minigrid.wrappers.RGBImgObsWrapper(env,tile_size=8)
#env = ImperviousToLava(env,novelty_episode=500000)

dv2.train(env, config)
