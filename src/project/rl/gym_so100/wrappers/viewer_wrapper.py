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



from __future__ import annotations

import gymnasium as gym
import mujoco
import mujoco.viewer


class PassiveViewerWrapper(gym.Wrapper):


    def __init__(
        self,
        env: gym.Env,
        *,
        show_left_ui: bool = False,
        show_right_ui: bool = False,
        show_joint_axes: bool = False,
        joint_axes_size: float = 0.05,
    ) -> None:
        
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
       
        observation, info = self.env.reset(**kwargs)
        self._viewer.sync()
        return observation, info

    def step(self, action):  # type: ignore[override]
        
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._viewer.sync()
        return observation, reward, terminated, truncated, info

    def close(self) -> None:  # type: ignore[override]
        

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

        if hasattr(self, "_viewer"):
            try:  # noqa: SIM105
                self._viewer.close()
            except Exception:
                pass
