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
# Gym SO100 环境包装器模块
# ================================================================
# 提供 SO100 机械臂的各种 Gym 环境包装器
# ================================================================

from gym_so100.wrappers.factory import (
    make_base_so100_env,
    make_so100_env_for_rollout,
    wrap_env,
)
from gym_so100.wrappers.hil_wrappers import (
    EEActionWrapper,
    GripperPenaltyWrapper,
    InputsControlWrapper,
    ResetDelayWrapper,
)
from gym_so100.wrappers.intervention_utils import (
    GamepadController,
    GamepadControllerHID,
    InputController,
    KeyboardController,
    load_controller_config,
)
from gym_so100.wrappers.viewer_wrapper import PassiveViewerWrapper

__all__ = [
    # 工厂函数
    "wrap_env",
    "make_base_so100_env",
    "make_so100_env_for_rollout",
    # HIL 包装器
    "EEActionWrapper",
    "GripperPenaltyWrapper",
    "InputsControlWrapper",
    "ResetDelayWrapper",
    # 查看器包装器
    "PassiveViewerWrapper",
    # 干预控制器
    "InputController",
    "KeyboardController",
    "GamepadController",
    "GamepadControllerHID",
    "load_controller_config",
]
