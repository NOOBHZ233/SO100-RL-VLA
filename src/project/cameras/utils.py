import platform
from typing import cast

from .camera import Camera
from .camera_config import CameraConfig, Cv2Rotation

'''
注册表+BaseClass,BaseClass的实例化main函数

'''

def make_camera_from_config(camera_configs :dict[str,CameraConfig])-> dict[str,Camera]:
    cameras :dict[str,Camera]
    for key, cfg in camera_configs.items():
        # TODO(Steven): Consider just using the make_device_from_device_class for all types
        if cfg.type == "opencv":
            from .opencv.opencv_camera import OpenCVCamera

            cameras[key] = OpenCVCamera(cfg)

        elif cfg.type == "intelrealsense":
            from .realsense.realsense_camera import RealSenseCamera

            cameras[key] = RealSenseCamera(cfg)

        else:
            raise ValueError(f"Invoild camera type , got{cfg.type}")
        
def get_cv2_rotation(rotation: Cv2Rotation) -> int | None:
    import cv2  # type: ignore  # TODO: add type stubs for OpenCV

    if rotation == Cv2Rotation.ROTATE_90:
        return int(cv2.ROTATE_90_CLOCKWISE)
    elif rotation == Cv2Rotation.ROTATE_180:
        return int(cv2.ROTATE_180)
    elif rotation == Cv2Rotation.ROTATE_270:
        return int(cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return None


def get_cv2_backend() -> int:
    import cv2

    if platform.system() == "Windows":
        return int(cv2.CAP_MSMF)  # Use MSMF for Windows instead of AVFOUNDATION
    # elif platform.system() == "Darwin":  # macOS
    #     return cv2.CAP_AVFOUNDATION
    else:  # Linux and others
        return int(cv2.CAP_ANY)