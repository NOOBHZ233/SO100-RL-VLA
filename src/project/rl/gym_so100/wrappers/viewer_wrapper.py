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

        # Launch interactive viewer
        # Expose model and data from unwrapped environment to ensure we can
        # access underlying MuJoCo objects even if other wrappers are applied
        self._viewer = mujoco.viewer.launch_passive(
            env.unwrapped.model,
            env.unwrapped.data,
            # show_left_ui=show_left_ui,
            # show_right_ui=show_right_ui,
        )

        # Configure joint axis visualization
        if show_joint_axes:
            self._viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE  # Show site coordinate frame to see gripper_site axes
            # Note: MuJoCo 3.4.0 does not have frame_size property; axis size is automatically determined by camera view
        else:
            # Hide all axes
            self._viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE

        # Ensure first frame is rendered
        self._viewer.sync()

    # ==================== Gym API Overrides ====================

    def reset(self, **kwargs):  # type: ignore[override]
       
        observation, info = self.env.reset(**kwargs)
        self._viewer.sync()
        return observation, info

    def step(self, action):  # type: ignore[override]
        
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._viewer.sync()
        return observation, reward, terminated, truncated, info

    def close(self) -> None:  # type: ignore[override]
        

        # 1. Clean up renderer managed by wrapped environment (if any)
        base_env = self.env.unwrapped  # type: ignore[attr-defined]
        if hasattr(base_env, "_viewer"):
            viewer = base_env._viewer
            if viewer is not None and hasattr(viewer, "close") and callable(viewer.close):
                try:  # noqa: SIM105
                    viewer.close()
                except Exception:
                    # Ignore errors from older MuJoCo versions or already released contexts
                    pass
            # Prevent underlying env from trying to close it again
            base_env._viewer = None

        # 2. Close passive viewer launched by this wrapper
        try:  # noqa: SIM105
            self._viewer.close()
        except Exception:  # pragma: no cover
            # Defensive handling: avoid propagating viewer close errors
            pass

        # 3. Let wrapped environment perform its own cleanup
        self.env.close()

    def __del__(self):

        if hasattr(self, "_viewer"):
            try:  # noqa: SIM105
                self._viewer.close()
            except Exception:
                pass
