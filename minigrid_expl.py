import gym
import gym_minigrid
from gym import ObservationWrapper

import dreamerv2.api as dv2

        
config = dv2.defaults.update({
    # 'logdir': '/srv/essa-lab/flash3/jballoch6/logs/dreamer',
    'logdir': '/home/balloch/logs/dreamer',
    'log_every': 1e3,
    'train_every': 10,
    'prefill': 1e5,
    'actor_ent': 3e-3,
    'loss_scales.kl': 1.0,
    'discount': 0.99,
# exploration stuff
    'expl_behavior':'Plan2Explore',
#  disag_target: stoch
#  disag_log: False
#  disag_models: 10
#  disag_offset: 1
#  disag_action_cond: True
#  expl_model_loss: kl
}).parse_flags()

#env = gym.make('MiniGrid-LavaShortcutMaze8x8-v0')
env = gym.make('MiniGrid-DoorKey-16x16-v0')
env = gym_minigrid.wrappers.RGBImgObsWrapper(env,tile_size=8)
#env = DoorKeyChange(env,novelty_episode=50000)
#env = ImperviousToLava(env,novelty_episode=50000)

dv2.train(env, config)
