from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation


class GridWorldEnv(gym.Env):

    def __init__(self, size: int = 5):
        # 方形网格的大小
        self.size = size

        # 定义智能体和目标位置，在 'reset' 中随机选择，在 'step' 中更新
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # 观测值是包含智能体和目标位置的字典。
        # 每个位置都编码为 {0， ...， 'size'-1}^2 的元素
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # 动作空间有 4 个动作，分别对应 “right”、“up”、“left”、“down”
        self.action_space = gym.spaces.Discrete(4)
        # 此字典将抽象动作映射到网格上的方向
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }

    # 获取观测值
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    # 获取智能体和目标之间的曼哈顿距离
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # 我们需要以下行来重置 self.np_random 随机化种子
        super().reset(seed=seed)

        # 随机生成智能体的位置
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # 随机生成目标的位置，直到它与智能体的位置不一致
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # 将作（{0,1,2,3} 的元素）映射到网格上的动作方向
        direction = self._action_to_direction[action]
        # 使用 'np.clip' 来确保智能体不会离开网格边界
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # 当且仅当代理已到达目标时，环境才完成
        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = False
        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

if __name__ == '__main__':
    n_episodes = 1000

    env = GridWorldEnv()
    # obs, info = env.reset()
    # print('obs: ', obs)
    # print('info: ', info)

    env = FlattenObservation(env)
    # obs, info = env.reset()
    # print('obs: ', obs)
    # print('info: ', info)

    for episode in range(n_episodes):
        obs, info = env.reset()
        print(f'********回合{episode}开始********')
        print('obs: ', obs)
        print('info: ', info)
        done = False
        step_count = 0

        while not done:
            action = env.action_space.sample() # 随机策略动作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or reward
            step_count += 1

        print(f'回合执行步数: {step_count}')
        print(f'********回合{episode}结束********\n')