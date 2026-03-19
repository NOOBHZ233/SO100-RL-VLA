#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ================================================================
# MuJoCo 机械臂 Gym 环境基类
# ================================================================
# 提供 SO100 等 6DOF 机械臂的 Gymnasium 环境接口
# 包含操作空间控制、观察空间、动作空间等核心功能
# ================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from gym_so100.controllers import opspace

# 夹爪最大控制命令值 (对应完全闭合)
MAX_GRIPPER_COMMAND = 255

#类被创建后无法被修改
@dataclass(frozen=True)
class GymRenderingSpec:
    """渲染配置参数

    Attributes:
        height: 图像高度 (像素)
        width: 图像宽度 (像素)
        camera_id: 相机ID (-1表示默认相机)
        mode: 渲染模式 ("rgb_array" 或 "human")
    """
    height: int = 128
    width: int = 128
    camera_id: str | int = -1
    mode: Literal["rgb_array", "human"] = "rgb_array"


class MujocoGymEnv(gym.Env):
    """MuJoCo 环境 Gym 接口基类

    提供 MuJoCo 物理仿真与 Gymnasium 标准接口之间的桥接。
    处理模型加载、渲染、物理步进等核心功能。
    """

    def __init__(
        self,
        xml_path: Path,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
    ):
        """初始化 MuJoCo 环境

        Args:
            xml_path: MuJoCo XML 文件路径
            seed: 随机种子
            control_dt: 控制周期 (秒)，即每次 action 的间隔
            physics_dt: 物理仿真步长 (秒)，MuJoCo 内部 timestep
            render_spec: 渲染配置
        """
        # 从 XML 加载 MuJoCo 模型
        self._model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        # 设置离屏渲染尺寸
        self._model.vis.global_.offwidth = render_spec.width
        self._model.vis.global_.offheight = render_spec.height
        # 创建仿真数据
        self._data = mujoco.MjData(self._model)
        # 设置物理仿真时间步长
        self._model.opt.timestep = physics_dt
        # 控制周期
        self._control_dt = control_dt
        # 每个控制周期内的物理子步数
        self._n_substeps = int(control_dt // physics_dt)
        # 随机数生成器
        self._random = np.random.RandomState(seed)
        # 渲染器 (延迟初始化)
        self._viewer: Optional[mujoco.Renderer] = None
        self._render_specs = render_spec

    def render(self):
        """渲染当前帧

        Returns:
            RGB 图像数组，如果没有渲染器则返回 None
        """
        if self._viewer is None:
            # 对于 human 模式，查看器由外部 wrapper 管理
            return None

        self._viewer.update_scene(self._data, camera=self._render_specs.camera_id)
        return self._viewer.render()

    def close(self) -> None:
        """释放图形资源

        兼容 MuJoCo < 2.3.0 (没有 close() 方法)
        """
        viewer = self._viewer
        if viewer is None:
            return

        if hasattr(viewer, "close") and callable(viewer.close):
            try:  # noqa: SIM105
                viewer.close()
            except Exception:
                # 忽略已释放的 OpenGL 上下文或旧版本 MuJoCo 的错误
                pass

        self._viewer = None

    # ==================== 属性访问器 ====================

    @property
    def model(self) -> mujoco.MjModel:
        """获取 MuJoCo 模型"""
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        """获取 MuJoCo 仿真数据"""
        return self._data

    @property
    def control_dt(self) -> float:
        """获取控制周期"""
        return self._control_dt

    @property
    def physics_dt(self) -> float:
        """获取物理仿真步长"""
        return self._model.opt.timestep

    @property
    def random_state(self) -> np.random.RandomState:
        """获取随机数生成器"""
        return self._random


class SO100GymEnv(MujocoGymEnv):
    """SO100 机械臂环境基类

    提供 SO100 6DOF 机械臂 + 内置夹爪的 Gym 环境接口。
    支持操作空间控制、图像观察等。

    SO100 关节配置:
    - Joint 0: shoulder_pan (肩部旋转)
    - Joint 1: shoulder_lift (肩部抬升)
    - Joint 2: elbow_flex (肘部弯曲)
    - Joint 3: wrist_flex (手腕弯曲)
    - Joint 4: wrist_roll (手腕旋转)
    - Joint 5: gripper (夹爪开合)
    """

    def __init__(
        self,
        xml_path: Path | None = None,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(), 
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        home_position: np.ndarray = np.asarray((0.0, -0.8, 1.5, 0.0, 0.0, 0.0)),  
        cartesian_bounds: np.ndarray = np.asarray([[-0.4, -0.4,  -0.4], [0.4, 0.4, 0.4]]),  
    ):
        """初始化 SO100 环境

        Args:
            xml_path: MuJoCo XML 文件路径 (None 则使用默认场景)
            seed: 随机种子
            control_dt: 控制周期 (秒)
            physics_dt: 物理仿真步长 (秒)
            render_spec: 渲染配置
            render_mode: 渲染模式 ("rgb_array" 或 "human")
            image_obs: 是否使用图像观察
            home_position: 机械臂 home 位置 (6个关节角度)
                - 默认: [0, 0, 0, 0, 0, 半开]
            cartesian_bounds: 笛卡尔空间边界 [min, max]
        """
        # 默认场景文件路径（使用 so_arm100.xml）
        if xml_path is None:
            xml_path = Path(__file__).parent / "model" / "scene.xml"

        super().__init__(
            xml_path=xml_path,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
        )

        # Home 位置和笛卡尔空间边界
        self._home_position = home_position
        self._cartesian_bounds = cartesian_bounds

        # 元数据
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self.image_obs = image_obs

        # ==================== 设置相机 ====================
        # SO100 有前置相机和腕部相机
        camera_name_1 = "front"
        camera_name_2 = "gripper_cam"
        camera_id_1 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name_1)
        camera_id_2 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name_2)
        self.camera_id = (camera_id_1, camera_id_2)

        # ==================== 缓存机器人 ID ====================
        # SO100 前5个关节（控制末端执行器）的 DOF IDs（与 so_arm100.xml 匹配）
        so100_arm_joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
        self._so100_arm_dof_ids = np.asarray([self._model.joint(name).id for name in so100_arm_joint_names])
        # SO100 执行器 IDs（与 so_arm100.xml 中的执行器名称匹配）
        so100_arm_actuator_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
        self._so100_arm_ctrl_ids = np.asarray([self._model.actuator(name).id for name in so100_arm_actuator_names])
        # 夹爪关节 DOF ID 和执行器 ID
        self._gripper_joint_name = "Jaw"
        self._gripper_dof_id = self._model.joint(self._gripper_joint_name).id
        self._gripper_ctrl_id = self._model.actuator("Jaw").id
        # 末端执行器站点 ID
        # 在 so_arm100.xml 中查找是否有对应的末端站点，或者使用其他方式获取
        # 首先检查是否有名为 gripper_site 的站点
        try:
            self._gripper_site_id = self._model.site("gripper_site").id
        except Exception:
            # 如果没有，我们需要找到夹爪的几何中心或者其他合适的参考点
            # 这里我们可以使用 Jaw 关节或 Fixed_Jaw 身体的位置
            # 或者我们可以添加一个站点到 so_arm100.xml 中
            # 暂时使用 Fixed_Jaw 身体的位置作为替代
            # 注意：这是一个临时解决方案，最好在 so_arm100.xml 中添加一个 gripper_site
            print("警告: 未找到 gripper_site 站点，将使用默认站点")
            # 检查是否有其他可用的站点
            if self._model.nsite > 0:
                self._gripper_site_id = 0
            else:
                # 如果没有站点，我们需要创建一个临时的或使用其他方法
                # 这里我们使用一个虚拟值，可能会导致错误
                self._gripper_site_id = -1

        # ==================== 设置观察空间和动作空间 ====================
        self._setup_observation_space()
        self._setup_action_space()

        # 初始化渲染器 (仅在 rgb_array 模式下)
        # human 模式会由 PassiveViewerWrapper 创建 launch_passive 查看器
        if self.render_mode == "rgb_array":
            self._viewer = mujoco.Renderer(self.model, height=render_spec.height, width=render_spec.width)
        else:
            self._viewer = None

    def _setup_observation_space(self):
        base_obs = {
            "joint_pose": spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32),
            "gripper_pose": spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        }

        if self.image_obs:
            base_obs["pixels"] = spaces.Dict(
                {
                    "front": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self._render_specs.height, self._render_specs.width, 3),
                        dtype=np.uint8,
                    ),
                    "wrist": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self._render_specs.height, self._render_specs.width, 3),
                        dtype=np.uint8,
                    ),
                }
            )

        self.observation_space = spaces.Dict(base_obs)

    def _setup_action_space(self):
        """设置 SO100 环境的动作空间
        末端执行器的XYZ 位置
        加爪开闭
    
        """
        self.action_space = spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset_robot(self):
        """将机械臂重置到 home 位置"""
        # 设置关节位置（前5个关节+夹爪）
        self._data.qpos[self._so100_arm_dof_ids] = self._home_position[:5]
        self._data.qpos[self._gripper_dof_id] = self._home_position[5]
        # 清零关节速度
        self._data.qvel[self._so100_arm_dof_ids] = 0.0
        self._data.qvel[self._gripper_dof_id] = 0.0
        # 清零关节控制力矩
        self._data.ctrl[self._so100_arm_ctrl_ids] = 0.0
        self._data.ctrl[self._gripper_ctrl_id] = 0.0
        # 前向运动学更新
        mujoco.mj_forward(self._model, self._data)

    def apply_action(self, action):
        """应用动作到机械臂

        Args:
            action: 4维动作向量
                   - 4维: [dx, dy, dz, grasp_command]
                   - dx,dy,dz: 位置增量 (缩放后约 ±0.05m)
                   - grasp_command: 夹爪控制 [-1, 1] (-1=闭合, 1=打开)
        """
        if len(action) == 4:
            x, y, z, grasp_command = action
        else:
            print(action)
            raise ValueError("dim of action is INVAID")
        # ==================== 计算目标位置 ====================
        current_pos = self._data.sensor("so100/gripper_site_pos").data
        dpos = np.asarray([x, y, z])
        # print(dpos)
        # print(current_pos)
        target_pos = np.clip(current_pos + dpos, *self._cartesian_bounds)
        # print(target_pos)
        # ==================== 更新夹爪开合度 ====================
        gripper_range = [-0.2, 2.0]
        target_gripper_pos = ((grasp_command + 1.0) / 2.0) * (gripper_range[1] - gripper_range[0]) + gripper_range[0]
        target_gripper_pos = np.clip(target_gripper_pos, *gripper_range)

        # ==================== 操作空间控制 ====================
        for _ in range(self._n_substeps):
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._gripper_site_id,
                dof_ids=self._so100_arm_dof_ids,
                pos=target_pos,
                ori=None,
                joint=self._home_position[:5],
                gravity_comp=True,
                nullspace_stiffness=2,
                pos_gains=(300.0, 300.0, 300.0),
            )
            self._data.ctrl[self._so100_arm_ctrl_ids] = tau
            # print(tau)
            # ==================== 夹爪控制 (PD控制) ====================
            kp_gripper = 8.0
            kd_gripper = 2.0
            current_gripper_pos = self._data.qpos[self._gripper_dof_id]
            current_gripper_vel = self._data.qvel[self._gripper_dof_id]
            gripper_tau = kp_gripper * (target_gripper_pos - current_gripper_pos) - kd_gripper * current_gripper_vel
            gripper_tau = np.clip(gripper_tau, -20.0, 20.0)
            self._data.ctrl[self._gripper_ctrl_id] = gripper_tau

            mujoco.mj_step(self._model, self._data)

    def get_joint_pose(self):
        return self.data.qpos[self._so100_arm_dof_ids].astype(np.float32)

    def get_robot_state(self):
        arm_qpos = self.data.qpos[self._so100_arm_dof_ids].astype(np.float32)
        gripper_pose = self.get_gripper_pose()
        gripper_pose = np.clip(gripper_pose, 0.0, 1.0)
        return np.concatenate([arm_qpos, gripper_pose])

    def render(self):
        """渲染环境并返回相机帧

        Returns:
            [front_view]: 前置相机的 RGB 图像 (如果是 rgb_array 模式)
        """
        if self._viewer is None:
            # human 模式下查看器由 PassiveViewerWrapper 管理
            return []

        rendered_frames = []
        for cam_id in self.camera_id:
            self._viewer.update_scene(self.data, camera=cam_id)
            rendered_frames.append(self._viewer.render())
        return rendered_frames

    def get_gripper_pose(self):
        """获取夹爪当前开合度

        Returns:
            归一化的夹爪位置 [0, 1]
            0 = 完全闭合, 1 = 完全打开
        """
        # SO100 夹爪使用单独的关节ID
        gripper_joint_pos = self._data.qpos[self._gripper_dof_id]
        # 映射关节范围 [-0.2, 2.0] 到 [0, 1]
        gripper_min = -0.2
        gripper_max = 2.0
        gripper_normalized = (gripper_joint_pos - gripper_min) / (gripper_max - gripper_min)
        gripper_normalized = np.clip(gripper_normalized, 0.0, 1.0)
        return np.array([gripper_normalized], dtype=np.float32)
