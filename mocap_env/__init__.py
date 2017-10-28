from gym.envs.registration import register
from .gym_mocap_walker import RoboschoolMocapHumanoid

register(
    id='RoboschoolMocapHumanoid-v1',
    entry_point='mocap_env:RoboschoolMocapHumanoid',
    max_episode_steps=1000,
    reward_threshold=3500.0
)