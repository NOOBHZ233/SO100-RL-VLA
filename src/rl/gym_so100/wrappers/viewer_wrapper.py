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
# 被动查看器包装器
# ================================================================
# 自动启动 MuJoCo 被动查看器的 Gym 包装器
# 适用于 SO100 机械臂环境
# ================================================================

from __future__ import annotations

import gymnasium as gym
import mujoco
import mujoco.viewer


class PassiveViewerWrapper(gym.Wrapper):
    """Gym 包装器，自动打开 MuJoCo 被动查看器

    包装器在环境创建后立即启动 MuJoCo 被动模式的查看器，
    用户无需手动使用 mujoco.viewer.launch_passive 或上下文管理器。

    查看器在每次 reset 和 step 调用后保持同步，
    环境关闭或删除时自动关闭。
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        show_left_ui: bool = False,
        show_right_ui: bool = False,
        show_joint_axes: bool = False,
        joint_axes_size: float = 0.05,
    ) -> None:
        """初始化被动查看器包装器

        Args:
            env: 要包装的 Gym 环境
            show_left_ui: 是否显示左侧 UI 面板
            show_right_ui: 是否显示右侧 UI 面板
            show_joint_axes: 是否显示关节坐标轴
            joint_axes_size: 关节坐标轴大小（米）
        """
        super().__init__(env)

        # 启动交互式查看器
        # 从 unwrapped 环境暴露 model 和 data，确保即使应用了其他包装器
        # 也能操作基础的 MuJoCo 对象
        self._viewer = mujoco.viewer.launch_passive(
            env.unwrapped.model,
            env.unwrapped.data,
            # show_left_ui=show_left_ui,
            # show_right_ui=show_right_ui,
        )

        # 配置关节坐标轴可视化
        if show_joint_axes:
            self._viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE  # 显示site坐标系，以便能看到gripper_site的坐标轴
            # 注意：MuJoCo 3.4.0中没有frame_size属性，坐标轴大小由相机视角自动决定
        else:
            # 隐藏所有坐标轴
            self._viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE

        # 确保渲染第一帧
        self._viewer.sync()

    # ==================== Gym API 重写 ====================

    def reset(self, **kwargs):  # type: ignore[override]
        """重置环境并同步查看器

        Returns:
            observation: 环境观察
            info: 额外信息字典
        """
        observation, info = self.env.reset(**kwargs)
        self._viewer.sync()
        return observation, info

    def step(self, action):  # type: ignore[override]
        """执行一步并同步查看器

        Args:
            action: 动作

        Returns:
            observation: 环境观察
            reward: 奖励
            terminated: 是否终止
            truncated: 是否截断
            info: 额外信息
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._viewer.sync()
        return observation, reward, terminated, truncated, info

    def close(self) -> None:  # type: ignore[override]
        """关闭被动查看器和底层 Gym 环境

        MuJoCo 的 Renderer 在较新版本 (>= 2.3.0) 才有 close() 方法。
        当使用旧版本 MuJoCo 时，env.unwrapped._viewer 中的渲染器实例
        不提供此方法，导致环境关闭时 AttributeError。

        为了保持版本兼容性：
        1. 仅当渲染器暴露 close 方法时才手动清理
        2. 从环境中移除引用，避免后续 env.close() 调用失败
        3. 关闭此包装器的被动查看器句柄
        4. 最后将 close() 调用转发给包装的环境，释放其他资源
        """

        # 1. 清理包装环境管理的渲染器（如果有）
        base_env = self.env.unwrapped  # type: ignore[attr-defined]
        if hasattr(base_env, "_viewer"):
            viewer = base_env._viewer
            if viewer is not None and hasattr(viewer, "close") and callable(viewer.close):
                try:  # noqa: SIM105
                    viewer.close()
                except Exception:
                    # 忽略来自旧版本 MuJoCo 或已释放上下文的错误
                    pass
            # 防止底层 env 尝试再次关闭它
            base_env._viewer = None

        # 2. 关闭此包装器启动的被动查看器
        try:  # noqa: SIM105
            self._viewer.close()
        except Exception:  # pragma: no cover
            # 防御性处理：避免传播查看器关闭错误
            pass

        # 3. 让包装的环境执行自己的清理
        self.env.close()

    def __del__(self):
        """析构函数，确保查看器被关闭

        close() 可能在解释器关闭时抛出异常；需要保护
        """
        if hasattr(self, "_viewer"):
            try:  # noqa: SIM105
                self._viewer.close()
            except Exception:
                pass
