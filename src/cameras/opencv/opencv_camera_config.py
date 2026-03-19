from dataclasses import dataclass
from pathlib import Path
import draccus

from ..camera_config import ColorMode,CameraConfig,Cv2Rotation

@CameraConfig.register_subclass("opencv")
@dataclass
class OpenCVCameraConfig(CameraConfig):
    index_or_path: int | Path
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    fourcc: str | None = None

    def __post_init__(self) ->None:
        if self.color_mode not in (ColorMode.BGR,ColorMode.RGB):
            raise ValueError(f"color_mode is expected as BGR or RGB")
        
        if self.rotation not in Cv2Rotation:
            raise ValueError(f"rotation is expected as Cv2Rotation")
        
        if self.fourcc is not None and (not isinstance(self.fourcc, str) or len(self.fourcc) != 4):
            raise ValueError(
                f"`fourcc` must be a 4-character string (e.g., 'MJPG', 'YUYV'), but '{self.fourcc}' is provided."
            )