import importlib
import importlib.metadata
import importlib.util
import logging
from typing import Any

from draccus.choice_types import ChoiceRegistry


"""
检查包是否已安装，并获取其版本号

该函数通过检查模块规范(module spec)来确认包是否存在,避免错误地导入
本地同名目录。特别处理了 torch 开发版本的情况。

Args:
    pkg_name: pip 安装时的包名（例如 "python-can")
    import_name: 实际导入时使用的模块名（例如 "can")
                如果未提供，则默认使用 pkg_name
    return_version: 是否返回版本字符串

Returns:
    如果 return_version=True,返回 (是否可用, 版本号) 元组
    如果 return_version=False,仅返回布尔值表示是否可用

Examples:
    >>> is_package_available("numpy")
    True
    >>> is_package_available("python-can", "can")
    True
    >>> is_package_available("torch", return_version=True)
    (True, "2.1.0+cu121")
"""
def is_package_available(
  pkg_name:str | None = None , import_name:str | None = None , return_version:bool = False
):
    if import_name == None:
        import_name = pkg_name
    # 使用 import_name 检查模块规范是否存在
    package_exists = importlib.util.find_spec(import_name) is not None
    package_version = "N/A"

    if package_exists:
        try:
           package_version = importlib.metadata.version(pkg_name) 
        except importlib.metadata.PackageNotFoundError:
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(import_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                except ImportError:
                    package_exists = False
            else:
                package_exists = False

        logging.debug(f"Detected {pkg_name} version: {package_version}")
    
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


_transformers_available = is_package_available("transformers")
_peft_available = is_package_available("peft")
_scipy_available = is_package_available("scipy")
_reachy2_sdk_available = is_package_available("reachy2_sdk")
_can_available = is_package_available("python-can", "can")
 
