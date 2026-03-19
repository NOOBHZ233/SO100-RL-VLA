import logging
from dataclasses import asdict , dataclass
from pprint import pformat

import draccus

from ..src.cameras.opencv.opencv_camera_config import OpenCVCameraConfig
from ..src.cameras.realsense.realsense_camera_config import RealSenseCameraConfig

from ..src.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    so100_follower,
)

from ..src.teleoperators import(
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
    so100_leader,
)

from ..src.utils.utils import init_logging

@dataclass
class CalibrateConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig

    def __post_init__(self)->None :
        if bool(self.teleop) == bool(self.robot):
            raise ValueError(f"teleop and robot shoudle be calibrated seq")
        
        self.device = self.teleop if self.teleop else self.robot
        
@draccus.wrap()
def calibrate(config:CalibrateConfig):
    init_logging()
    logging.info(pformat(asdict(config)))

    if isinstance(config.device,TeleoperatorConfig):
        device = make_teleoperator_from_config(config)

    if isinstance(config.device,RobotConfig):
        device = make_robot_from_config(config)

    device.connect(calibrate=False)

    try:
        device.calibrate()
    finally:
        device.disconnect()

def main():
    calibrate()

if __name__ == "__main__":
    main()


