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


import json
from pathlib import Path


def load_controller_config(controller_name: str, config_path: str | None = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent / "controller_config.json"

    with open(config_path) as f:
        config = json.load(f)

    controller_config = config[controller_name] if controller_name in config else config["default"]

    if controller_name not in config:
        print(f"Controller {controller_name} not found in config. Using default configuration.")

    return controller_config


class InputController:

    def __init__(self, x_step_size=0.01, y_step_size=0.01, z_step_size=0.01):

        self.x_step_size = x_step_size
        self.y_step_size = y_step_size
        self.z_step_size = z_step_size
        self.running = True
        self.episode_end_status = None  
        self.intervention_flag = False
        self.open_gripper_command = False
        self.close_gripper_command = False

    def start(self):
        pass

    def stop(self):

        pass

    def reset(self):

        pass

    def get_deltas(self):
       
        return 0.0, 0.0, 0.0

    def update(self):
        
        pass

    def __enter__(self):
       
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
       
        self.stop()

    def get_episode_end_status(self):
        status = self.episode_end_status
        self.episode_end_status = None  # 读取后重置
        return status


        return self.intervention_flag

    def gripper_command(self):
        
        if self.open_gripper_command == self.close_gripper_command:
            return "no-op"
        elif self.open_gripper_command:
            return "open"
        elif self.close_gripper_command:
            return "close"


class KeyboardController(InputController):
    
    def __init__(self, x_step_size=0.01, y_step_size=0.01, z_step_size=0.01, require_intervention_key=False):
    
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.require_intervention_key = require_intervention_key
        self.key_states = {
            "forward_x": False,
            "backward_x": False,
            "forward_y": False,
            "backward_y": False,
            "forward_z": False,
            "backward_z": False,
            "success": False,
            "failure": False,
            "intervention": not require_intervention_key,  # 默认激活（如果不需要干预键）
            "rerecord": False,
        }
        self.listener = None

    def start(self):
        
        from pynput import keyboard

        def on_press(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = True
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = True
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = True
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = True
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = True
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = True
                elif key == keyboard.Key.ctrl_r:
                    self.open_gripper_command = True
                elif key == keyboard.Key.ctrl_l:
                    self.close_gripper_command = True
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = True
                    self.episode_end_status = "success"
                elif key == keyboard.Key.esc:
                    self.key_states["failure"] = True
                    self.episode_end_status = "failure"
                elif key == keyboard.Key.space:
                    self.key_states["intervention"] = not self.key_states["intervention"]
                elif key == keyboard.Key.r:
                    self.key_states["rerecord"] = True
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = False
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = False
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = False
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = False
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = False
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = False
                elif key == keyboard.Key.ctrl_r:
                    self.open_gripper_command = False
                elif key == keyboard.Key.ctrl_l:
                    self.close_gripper_command = False
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

        print("键盘控制 (SO100 机械臂):")
        print("  方向键: XY 平面移动")
        print("  Shift / Shift_R: Z 轴移动")
        print("  右 Ctrl / 左 Ctrl: 打开/闭合夹爪")
        print("  Enter: 标记 episode 成功结束")
        print("  ESC: 标记 episode 失败结束")
        if self.require_intervention_key:
            print("  空格: 开始/停止干预模式")
        else:
            print("  控制模式: 直接激活 (无需空格)")
        print("  R: 重新记录 episode")

    def stop(self):

        
        if self.listener and self.listener.is_alive():
            self.listener.stop()

    def get_deltas(self):


        delta_x = delta_y = delta_z = 0.0

        if self.key_states["forward_x"]:
            delta_x += self.x_step_size
        if self.key_states["backward_x"]:
            delta_x -= self.x_step_size
        if self.key_states["forward_y"]:
            delta_y += self.y_step_size
        if self.key_states["backward_y"]:
            delta_y -= self.y_step_size
        if self.key_states["forward_z"]:
            delta_z += self.z_step_size
        if self.key_states["backward_z"]:
            delta_z -= self.z_step_size

        return delta_x, delta_y, delta_z

    def should_save(self):
       
        return self.key_states["success"] or self.key_states["failure"]

    def should_intervene(self):
       
        return self.key_states["intervention"]

    def reset(self):
       

        intervention_state = self.key_states.get("intervention", False)
        for key in self.key_states:
            if key != "intervention":
                self.key_states[key] = False
        self.key_states["intervention"] = intervention_state



