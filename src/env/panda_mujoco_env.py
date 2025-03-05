import time

import numpy as np
import mujoco
from mujoco import viewer
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


class PandaGraspEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, render_mode=None):
        super().__init__()

        # 加载Panda机械臂模型
        self.panda_xml_path = r"../../arm_description/panda/panda_scene.xml"
        self.model = mujoco.MjModel.from_xml_path(self.panda_xml_path)
        self.data = mujoco.MjData(self.model)

        # 添加目标方块
        self._add_target_box()

        # 渲染设置
        self.render_mode = render_mode
        if render_mode == 'human':
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # 动作空间：7个关节的力矩控制 + 夹爪开合
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # 观察空间：关节角度、末端位置、方块位置、夹爪状态
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(23,), dtype=np.float64
        )

        # 初始化目标位置
        self.target_pos = np.array([0.5, 0.0, 0.5])
        self.reset_target()

        # 训练参数
        self.max_steps = 500
        self.current_step = 0

    def _add_target_box(self):
        # 在MuJoCo模型中添加目标方块
        # body = """
        # <body name="target_box" pos="0.5 0 0.5">
        #     <geom type="box" size="0.05 0.05 0.05" rgba="1 0 0 1"/>
        # </body>
        # """
        panda_spec = mujoco.MjSpec.from_file(self.panda_xml_path)
        target_body = panda_spec.worldbody.add_body(name="target_box", pos=[0.5, 0.0, 0.5])
        target_body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05], rgba=[1, 0, 0, 1])
        self.model = panda_spec.compile()
        self.data = mujoco.MjData(self.model)

    def _get_obs(self):
        # 获取观察值
        return np.concatenate([
            self.data.qpos[:7],  # 关节角度
            self.data.qvel[:7],  # 关节速度
            self.data.body("target_box").xpos,  # 方块位置
            self.data.body("hand").xpos,  # 末端执行器位置
            self.data.qpos[7:9],  # 夹爪状态
            [self.data.time]  # 当前时间
        ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 重置模型
        mujoco.mj_resetData(self.model, self.data)

        # 随机初始化目标位置
        self.reset_target()

        # 随机初始化机械臂位置
        self.data.qpos[:7] = np.random.uniform(-0.5, 0.5, size=7)

        self.current_step = 0

        obs = self._get_obs()
        print('reset_obs:', obs)
        return obs, {}

    def reset_target(self):
        # 随机设置目标位置
        self.target_pos = np.random.uniform([0.3, -0.3, 0.4], [0.7, 0.3, 0.6])
        self.model.body("target_box").pos = self.target_pos
        # self.data.body("target_box").xpos = self.target_pos

    def step(self, action):
        # 应用动作
        ctrl = action * 100  # 缩放控制信号
        self.data.ctrl[:] = ctrl

        # 仿真步进
        mujoco.mj_step(self.model, self.data)

        # 获取观察
        obs = self._get_obs()
        print('current_step', self.current_step)
        print('step_obs:', obs)

        # 计算奖励
        reward = self._calculate_reward()

        # 终止条件
        terminated = self.current_step >= self.max_steps
        truncated = False

        self.current_step += 1

        if self.render_mode == 'human':
            self.viewer.sync()

        return obs, reward, terminated, truncated, {}

    def _calculate_reward(self):
        # 奖励函数组件
        end_effector_pos = self.data.body("hand").xpos
        target_pos = self.data.body("target_box").xpos
        grasp_success = self._check_grasp()

        # 距离奖励
        distance = np.linalg.norm(end_effector_pos - target_pos)
        distance_reward = 1.0 / (1.0 + distance)

        # 抓取成功奖励
        grasp_reward = 10.0 if grasp_success else 0.0

        # 控制惩罚
        control_penalty = -0.01 * np.sum(np.square(self.data.ctrl))

        return distance_reward + grasp_reward + control_penalty

    def _check_grasp(self):
        # 检查是否成功抓取
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = self.model.geom(contact.geom1).name
            geom2 = self.model.geom(contact.geom2).name
            if ("hand" in geom1 and "target_box" in geom2) or \
                    ("hand" in geom2 and "target_box" in geom1):
                return True
        return False


if __name__ == '__main__':
    # 环境验证
    env = PandaGraspEnv(render_mode='human')
    check_env(env)

    # 演示训练结果
    obs, _ = env.reset()
    # 演示
    obs, _ = env.reset()
    for _ in range(10000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
