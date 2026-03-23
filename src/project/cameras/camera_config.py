import abc
from dataclasses import dataclass
from enum import Enum

import draccus

class ColorMode(str,Enum):
    RGB = "rgb"
    BGR = "bgr"

class Cv2Rotation(int, Enum):
    NO_ROTATION = 0
    ROTATE_90 = 90
    ROTATE_180 = 180
    ROTATE_270 = -90

@dataclass
class CameraConfig(draccus.ChoiceRegistry,abc.ABC):
    fps: int | None = None
    width: int | None = None
    height: int | None = None

    @property
    def type(self)-> str :
        return str(self.get_choice_name(self.__class__))
