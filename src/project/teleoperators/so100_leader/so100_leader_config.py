from dataclasses import dataclass,field
from ..teleoperator_config import TeleoperatorConfig

@dataclass
class SO100LeaderConfig:
    port :str | None = None
    use_degrees: bool = False

@TeleoperatorConfig.register_subclass("so100_leader")
@dataclass
class SO100LeaderTeleoperatorConfig(TeleoperatorConfig,SO100LeaderConfig):
    pass
#参数足够，直接pass