import time
from stable_baselines3 import PPO
from src.env.panda_mujoco_env import PandaGraspEnv
from stable_baselines3.common.env_checker import check_env

# 环境验证
env = PandaGraspEnv(render_mode='human')
check_env(env)

# 创建PPO模型
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="../../logs/panda_mujoco_tensorboard/",
    batch_size=256,
    learning_rate=3e-4
)

# 训练模型
# model.learn(total_timesteps=1_000_000)
model.learn(total_timesteps=1_000)

# 保存模型
now = time.strftime('%m%d-%H%M%S', time.localtime())
model.save("../../models/panda_mujoco/PPO/" + now + "_model")