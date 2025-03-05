from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """使用状态作值 （q_values）空字典、学习率和 epsilon 初始化强化学习智能体。

        Args:
            env: 训练环境
            learning_rate: 学习率
            initial_epsilon: 初始 epsilon 值
            epsilon_decay: epsilon 衰减值
            final_epsilon: 最终 epsilon 值
            discount_factor: 用于计算 Q 值的折扣因子
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
       返回概率为 （1 - epsilon） 的最佳动作，否则返回概率为 epsilon 的随机动作，以确保探索。
        """
        # 概率为 epsilon 返回一个随机作来探索环境
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # 概率 （1 - Epsilon） 贪心探索
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """更新动作的 Q 值。"""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

if __name__ == '__main__':
    # 超参数
    learning_rate = 0.01
    n_episodes = 100_000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)  # 随着时间的推移减少探索
    final_epsilon = 0.1

    env = gym.make("Blackjack-v1", sab=False)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    agent = BlackjackAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    # 训练智能体
    for episode in tqdm(range(n_episodes)): # tqdm可视化训练进度
        obs, info = env.reset()
        done = False

        # 开始回合
        while not done:
            action = agent.get_action(obs) # 智能体策略
            next_obs, reward, terminated, truncated, info = env.step(action)

            # 更新智能体策略
            agent.update(obs, action, reward, terminated, next_obs)

            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon() # 探索衰减

    # 可视化训练过程
    # 回合数据处理
    def get_moving_avgs(arr, window, convolution_mode):
        return np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode=convolution_mode
        ) / window


    # 每 500 回合平滑处理
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)

    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "same"
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    plt.tight_layout()
    plt.show()