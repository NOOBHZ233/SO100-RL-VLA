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

from enum import Enum
from typing import TYPE_CHECKING, cast


from .teleoperator_config import TeleoperatorConfig


from project.teleoperators.teleoperator import Teleoperator


class TeleopEvents(Enum):
    """Shared constants for teleoperator events across teleoperators."""

    SUCCESS = "success"
    FAILURE = "failure"
    RERECORD_EPISODE = "rerecord_episode"
    IS_INTERVENTION = "is_intervention"
    TERMINATE_EPISODE = "terminate_episode"


def make_teleoperator_from_config(config :TeleoperatorConfig | None = None) ->Teleoperator:
    if config ==None:
        raise(f"config should be determinted,now got None")
    if config.type == "so100_leader":
        from so100_leader import SO100LeaderTeleoperatorConfig
        print(f"Init the teleoperator so100_leader")

        return SO100LeaderTeleoperatorConfig(config)
    else:
        raise ValueError(
            f"The config.type is not correct, got {config.type}"
        )