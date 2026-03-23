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

"""
导入工具模块

该模块提供了以下功能：
1. 检查 Python 包是否可用，并获取其版本
2. 根据配置类动态实例化设备对象
3. 自动发现和导入第三方 LeRobot 插件
"""

import importlib
import importlib.metadata
import importlib.util
import logging
from typing import Any

from draccus.choice_types import ChoiceRegistry


# ============================================================================
# 工具函数：检查包是否可用
# ============================================================================

def is_package_available(
    pkg_name: str, import_name: str | None = None, return_version: bool = False
) -> tuple[bool, str] | bool:
    """
    检查包是否已安装，并获取其版本号

    该函数通过检查模块规范（module spec）来确认包是否存在，避免错误地导入
    本地同名目录。特别处理了 torch 开发版本的情况。

    Args:
        pkg_name: pip 安装时的包名（例如 "python-can"）
        import_name: 实际导入时使用的模块名（例如 "can"）
                    如果未提供，则默认使用 pkg_name
        return_version: 是否返回版本字符串

    Returns:
        如果 return_version=True，返回 (是否可用, 版本号) 元组
        如果 return_version=False，仅返回布尔值表示是否可用

    Examples:
        >>> is_package_available("numpy")
        True
        >>> is_package_available("python-can", "can")
        True
        >>> is_package_available("torch", return_version=True)
        (True, "2.1.0+cu121")
    """
    if import_name is None:
        import_name = pkg_name

    # 使用 import_name 检查模块规范是否存在
    package_exists = importlib.util.find_spec(import_name) is not None
    package_version = "N/A"

    if package_exists:
        try:
            # 主要方法：通过 pkg_name 获取包版本
            package_version = importlib.metadata.version(pkg_name)

        except importlib.metadata.PackageNotFoundError:
            # 回退方法：仅针对 torch 处理包含 "dev" 的版本
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(import_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    # 只有版本包含 "dev" 时才接受这个回退值
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                except ImportError:
                    # 如果无法导入包，则认为不可用
                    package_exists = False
            else:
                # 对于非 torch 包，不尝试回退方法
                package_exists = False

        logging.debug(f"Detected {pkg_name} version: {package_version}")

    if return_version:
        return package_exists, package_version
    else:
        return package_exists


# ============================================================================
# 模块级变量：缓存常用包的可用性检查结果
# ============================================================================

# 这些变量在模块加载时计算，用于快速检查依赖包是否可用
_transformers_available = is_package_available("transformers")
_peft_available = is_package_available("peft")
_scipy_available = is_package_available("scipy")
_reachy2_sdk_available = is_package_available("reachy2_sdk")
_can_available = is_package_available("python-can", "can")


# ============================================================================
# 工厂函数：从配置类动态实例化设备
# ============================================================================

def make_device_from_device_class(config: ChoiceRegistry) -> Any:
    """
    根据配置对象动态实例化对应的设备类

    这是一个工厂函数，通过配置对象的模块路径和类名来定位并实例化
    相应的设备类（而非配置类）。设备类名通过移除配置类名末尾的
    'Config' 得到，然后在几个可能的模块中搜索设备类的实现。

    Args:
        config: 继承自 ChoiceRegistry 的配置对象

    Returns:
        实例化后的设备对象

    Raises:
        ValueError: 如果 config 不是 ChoiceRegistry 实例，或配置类名不以 'Config' 结尾
        TypeError: 如果设备类实例化失败
        ImportError: 如果无法找到设备类

    Examples:
        >>> config = MyDeviceConfig(param=1)
        >>> device = make_device_from_device_class(config)
        # 等价于直接调用 MyDevice(config)
    """
    # 验证输入类型
    if not isinstance(config, ChoiceRegistry):
        raise ValueError(f"Config should be an instance of `ChoiceRegistry`, got {type(config)}")

    # 获取配置类的元信息
    config_cls = config.__class__
    module_path = config_cls.__module__  # 例如: lerobot_teleop_mydevice.config_mydevice
    config_name = config_cls.__name__     # 例如: MyDeviceConfig

    # 推导设备类名（移除末尾的 "Config"）
    if not config_name.endswith("Config"):
        raise ValueError(f"Config class name '{config_name}' does not end with 'Config'")

    device_class_name = config_name[:-6]  # 例如: MyDeviceConfig -> MyDevice

    # 构建候选模块列表，在这些模块中搜索设备类
    parts = module_path.split(".")
    parent_module = ".".join(parts[:-1]) if len(parts) > 1 else module_path
    candidates = [
        parent_module,  # 例如: lerobot_teleop_mydevice
        parent_module + "." + device_class_name.lower(),  # 例如: lerobot_teleop_mydevice.mydevice
    ]

    # 处理 "config_xxx" 格式的模块名 -> 尝试替换为 "xxx"
    last = parts[-1] if parts else ""
    if last.startswith("config_"):
        candidates.append(".".join(parts[:-1] + [last.replace("config_", "")]))

    # 去重，同时保持顺序
    seen: set[str] = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    # 在候选模块中搜索并实例化设备类
    tried: list[str] = []
    for candidate in candidates:
        tried.append(candidate)
        try:
            module = importlib.import_module(candidate)
        except ImportError:
            continue

        # 检查模块中是否存在目标类
        if hasattr(module, device_class_name):
            cls = getattr(module, device_class_name)
            if callable(cls):
                try:
                    # 找到类，尝试实例化
                    return cls(config)
                except TypeError as e:
                    raise TypeError(
                        f"Failed to instantiate '{device_class_name}' from module '{candidate}': {e}"
                    ) from e

    # 所有候选模块都搜索失败
    raise ImportError(
        f"Could not locate device class '{device_class_name}' for config '{config_name}'. "
        f"Tried modules: {tried}. Ensure your device class name is the config class name without "
        f"'Config' and that it's importable from one of those modules."
    )


# ============================================================================
# 插件注册函数：发现并导入第三方插件
# ============================================================================

def register_third_party_plugins() -> None:
    """
    发现并导入第三方 LeRobot 插件，使其能够自动注册

    该函数使用 importlib.metadata 扫描环境中已安装的包（包括可编辑安装），
    查找以 'lerobot_robot_'、'lerobot_camera_'、'lerobot_teleoperator_' 或
    'lerobot_policy_' 开头的包，并导入它们。导入时插件会自动注册到相应注册表中。

    Note:
        插件导入失败不会中断程序，但会记录错误日志

    Examples:
        >>> register_third_party_plugins()
        Imported third-party plugin: lerobot_robot_myrobot
        Could not import third-party plugin: lerobot_camera_broken
    """
    # 定义插件包的前缀
    prefixes = ("lerobot_robot_", "lerobot_camera_", "lerobot_teleoperator_", "lerobot_policy_")
    imported: list[str] = []  # 成功导入的插件列表
    failed: list[str] = []    # 导入失败的插件列表

    def attempt_import(module_name: str):
        """尝试导入一个插件模块，记录结果"""
        try:
            importlib.import_module(module_name)
            imported.append(module_name)
            logging.info("Imported third-party plugin: %s", module_name)
        except Exception:
            logging.exception("Could not import third-party plugin: %s", module_name)
            failed.append(module_name)

    # 遍历所有已安装的包
    for dist in importlib.metadata.distributions():
        dist_name = dist.metadata.get("Name")
        if not dist_name:
            continue
        # 如果包名匹配插件前缀，尝试导入
        if dist_name.startswith(prefixes):
            attempt_import(dist_name)

    # 记录导入摘要
    logging.debug("Third-party plugin import summary: imported=%s failed=%s", imported, failed)
