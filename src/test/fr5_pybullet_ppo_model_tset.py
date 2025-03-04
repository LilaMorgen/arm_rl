from stable_baselines3 import A2C, PPO, DDPG, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from src.env.fr5_pybullet_env import FR5_Env
import time
import os
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure

env = FR5_Env(gui=True)
env.render()
model = PPO.load(r"../../models/fr5_pybullet/PPO/model94.zip")
# model = PPO.load(r"../../models/fr5_pybullet/PPO/1208-122633/best_model.zip")
# model = TD3.load(r"../../models/fr5_pybullet/TD3/1208-122633/best_model.zip")
# model = DDPG.load(r"../../models/fr5_pybullet/DDPG/1208-122633/best_model.zip")
test_num = 100  # 测试次数
success_num = 0  # 成功次数
print("测试次数：", test_num)
for i in range(test_num):
    state, _ = env.reset()
    done = False
    score = 0
    # time.sleep(3)

    while not done:
        # action = env.action_space.sample()     # 随机采样动作
        action, _ = model.predict(observation=state, deterministic=True)
        # print("state:",state)
        # print("action:",action)
        state, reward, done, _, info = env.step(action=action)
        score += reward
        # env.render()
        time.sleep(0.01)

    if info['is_success']:
        success_num += 1
    print("奖励：", score)
success_rate = success_num / test_num
print("成功率：", success_rate)
env.close()

