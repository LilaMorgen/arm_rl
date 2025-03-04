import time
from stable_baselines3 import PPO
from src.env.panda_mujoco_env import PandaGraspEnv
from stable_baselines3.common.env_checker import check_env

# 环境验证
env = PandaGraspEnv(render_mode='human')
check_env(env)

model = PPO.load(r"../../models/panda_mujoco/PPO/0301-190542_model.zip", env=env)

# 演示训练结果
obs, _ = env.reset()
for _ in range(10000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()