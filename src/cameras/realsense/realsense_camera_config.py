from ..camera_config import CameraConfig
from dataclasses import dataclass
from pathlib import Path
from ..camera_config import ColorMode , Cv2Rotation


@CameraConfig.register_subclass("realsense")
@dataclass
class RealSenseCameraConfig(CameraConfig):
    serial_number_or_name: str
    color_mode: ColorMode = ColorMode.RGB
    use_depth: bool = False
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1

    def __post_init__(self) ->None:
        if self.color_mode not in (ColorMode.BGR,ColorMode.RGB):
            raise ValueError(f"color_mode is expected as BGR or RGB")
        
        if self.rotation not in Cv2Rotation:
            raise ValueError(f"rotation is expected as Cv2Rotation")
        
        values = (self.fps, self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them."
            )