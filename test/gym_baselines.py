import gymnasium as gym

import sys
import os

sys.path.append(os.path.realpath("."))

import dumb_balatro

from stable_baselines3 import PPO

env = gym.make("DumbBalatro")

model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_dumb_balatro_tensorboard/",
)
model.learn(total_timesteps=100_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(10):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    print(reward)
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()
