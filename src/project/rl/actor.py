#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
Actor 服务器 - 分布式 HILSerl 机器人策略训练

本脚本实现了分布式 HILSerl 架构中的 Actor 组件。
Actor 负责在机器人环境中执行策略、收集经验数据，
并将转换数据发送给 Learner 服务器进行策略更新。

职责：
1. 在真实/模拟机器人环境中执行策略
2. 收集状态转换数据 (state, action, reward, next_state)
3. 支持人工干预 (Human-in-the-Loop)
4. 通过 gRPC 向 Learner 发送数据
5. 接收并加载更新后的策略参数

使用示例：

- 启动 Actor 服务器进行带人工干预的真实机器人训练：
```bash
python -m lerobot.rl.actor --config_path src/lerobot/configs/train_config_hilserl_so100.json
```

**注意**: Actor 需要连接到运行中的 Learner 服务器。启动 Actor 前请确保 Learner 已启动。

**注意**: 人工干预是 HILSerl 训练的关键。在训练过程中按下游戏手柄的右上角扳机键
          可以接管机器人控制权。初期应频繁干预，随着策略改善逐渐减少干预。

**完整工作流程**：
1. 使用 `lerobot-find-joint-limits` 确定机器人工作空间边界
2. 使用 `gym_manipulator.py` 的 record 模式录制示范数据
3. 使用 `crop_dataset_roi.py` 处理数据集并确定相机裁剪区域
4. 启动 Learner 服务器
5. 使用相同配置启动本 Actor 服务器
6. 使用人工干预指导策略学习

完整的 HILSerl 训练流程详见：
https://github.com/michel-aractingi/lerobot-hilserl-guide

数据流：
┌─────────────┐                    ┌─────────────┐
│   Actor     │                    │   Learner   │
│             │                    │             │
│ ┌─────────┐ │                    │ ┌─────────┐ │
│ │ Policy  │ │ ◄──参数更新──────── │ │   SAC   │ │
│ │ Inference│ │     gRPC Stream   │ │ Training │ │
│ └────┬────┘ │                    │ └────┬────┘ │
│      │      │                    │      │      │
│      ▼      │                    │      ▼      │
│ ┌─────────┐ │ ───Transitions───▶ │ ┌─────────┐ │
│ │ Robot   │ │     gRPC Stream   │ │ Replay  │ │
│ │  Env    │ │                    │ │ Buffer  │ │
│ └─────────┘ │                    │ └─────────┘ │
└─────────────┘                    └─────────────┘
"""

import logging
import os
import time
from functools import lru_cache
from queue import Empty

import grpc
import torch
from torch import nn
from torch.multiprocessing import Event, Queue

from project.cameras import opencv  # noqa: F401
from project.configs import parser
from project.configs.train import TrainRLServerPipelineConfig
from project.policies.factory import make_policy
from project.policies.sac.modeling_sac import SACPolicy
from project.processor import TransitionKey
from project.rl.process import ProcessSignalHandler
from project.rl.queue import get_last_item_from_queue
from project.robots import so100_follower  # noqa: F401
from project.teleoperators.utils import TeleopEvents
from project.transport import services_pb2, services_pb2_grpc
from project.transport.utils import (
    bytes_to_state_dict,
    grpc_channel_options,
    python_object_to_bytes,
    receive_bytes_in_chunks,
    send_bytes_in_chunks,
    transitions_to_bytes,
)
from project.utils.random_utils import set_seed
from project.utils.robot_utils import precise_sleep
from project.utils.transition import (
    Transition,
    move_state_dict_to_device,
    move_transition_to_device,
)
from project.utils.utils import (
    TimerManager,
    get_safe_torch_device,
    init_logging,
)

from .gym_manipulator import (
    create_transition,
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)

# ============================================================================
# 主入口点
# ============================================================================

@parser.wrap()
def actor_cli(cfg: TrainRLServerPipelineConfig):
    """
    Actor 命令行入口函数

    这是 Actor 服务器的入口点，负责：
    1. 解析并验证配置
    2. 初始化日志系统
    3. 连接到 Learner gRPC 服务器
    4. 启动三个通信线程/进程：
       - receive_policy: 接收更新后的策略参数
       - send_transitions: 发送转换数据
       - send_interactions: 发送交互统计数据
    5. 启动主执行循环 act_with_policy()

    Args:
        cfg: 训练配置对象，包含环境、策略和通信配置

    流程图：
        ┌─────────────────┐
        │ 解析配置参数     │
        └────────┬────────┘
                 ▼
        ┌─────────────────┐
        │ 初始化日志系统   │
        └────────┬────────┘
                 ▼
        ┌─────────────────┐
        │ 连接 Learner    │ ───失败──▶ 返回错误
        └────────┬────────┘
                 ▼
        ┌─────────────────┐
        │ 创建通信队列     │
        └────────┬────────┘
                 ▼
        ┌─────────────────┐
        │ 启动通信进程     │
        └────────┬────────┘
                 ▼
        ┌─────────────────┐
        │ 执行策略主循环   │
        └─────────────────┘
    """
    # 验证配置参数
    cfg.validate()

    # 决定是否显示进程 ID（多进程模式下需要）
    display_pid = False
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")  # 多进程使用 spawn 模式
        display_pid = True

    # 创建日志目录并初始化日志系统
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"actor_{cfg.job_name}.log")

    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Actor logging initialized, writing to {log_file}")

    # 创建信号处理器，用于优雅关闭
    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    # 创建 Learner gRPC 客户端
    learner_client, grpc_channel = learner_service_client(
        host=cfg.policy.actor_learner_config.learner_host,
        port=cfg.policy.actor_learner_config.learner_port,
    )

    # 建立与 Learner 的连接
    logging.info("[ACTOR] Establishing connection with Learner")
    if not establish_learner_connection(learner_client, shutdown_event):
        logging.error("[ACTOR] Failed to establish connection with Learner")
        return

    # 如果使用多进程模式，关闭主进程的 gRPC 通道
    # （子进程会创建自己的通道）
    if not use_threads(cfg):
        grpc_channel.close()
        grpc_channel = None

    logging.info("[ACTOR] Connection with Learner established")

    # 创建三个通信队列，用于与通信进程交换数据
    parameters_queue = Queue()      # 存储从 Learner 接收的策略参数
    transitions_queue = Queue()     # 存储待发送给 Learner 的转换数据
    interactions_queue = Queue()    # 存储待发送给 Learner 的统计数据

    # 根据配置选择并发实体（线程或进程）
    concurrency_entity = None
    if use_threads(cfg):
        from threading import Thread
        concurrency_entity = Thread
    else:
        from multiprocessing import Process
        concurrency_entity = Process

    # 创建接收策略参数的进程/线程
    receive_policy_process = concurrency_entity(
        target=receive_policy,
        args=(cfg, parameters_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    # 创建发送转换数据的进程/线程
    transitions_process = concurrency_entity(
        target=send_transitions,
        args=(cfg, transitions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    # 创建发送统计数据的进程/线程
    interactions_process = concurrency_entity(
        target=send_interactions,
        args=(cfg, interactions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    # 启动所有通信进程
    transitions_process.start()
    interactions_process.start()
    receive_policy_process.start()

    # 启动主执行循环（在当前进程中运行）
    act_with_policy(
        cfg=cfg,
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        transitions_queue=transitions_queue,
        interactions_queue=interactions_queue,
    )
    logging.info("[ACTOR] Policy process joined")

    # 清理资源：关闭队列
    logging.info("[ACTOR] Closing queues")
    transitions_queue.close()
    interactions_queue.close()
    parameters_queue.close()

    # 等待所有通信进程结束
    transitions_process.join()
    logging.info("[ACTOR] Transitions process joined")
    interactions_process.join()
    logging.info("[ACTOR] Interactions process joined")
    receive_policy_process.join()
    logging.info("[ACTOR] Receive policy process joined")

    logging.info("[ACTOR] join queues")
    transitions_queue.cancel_join_thread()
    interactions_queue.cancel_join_thread()
    parameters_queue.cancel_join_thread()

    logging.info("[ACTOR] queues closed")


# ============================================================================
# 核心算法函数
# ============================================================================

def act_with_policy(
    cfg: TrainRLServerPipelineConfig,
    shutdown_event: any,  # Event,
    parameters_queue: Queue,
    transitions_queue: Queue,
    interactions_queue: Queue,
):
    """
    在环境中执行策略交互（Actor 主循环）

    这是 Actor 的核心执行函数，负责：
    1. 初始化环境和策略
    2. 在环境中执行策略，收集交互数据
    3. 检测并处理人工干预
    4. 将转换数据推送到队列发送给 Learner
    5. Episode 结束时更新策略参数
    6. 发送统计数据到 Learner

    数据流：
        ┌─────────────────────────────────────────────────────────────┐
        │  观察状态 → 策略推断 → 执行动作 → 获得奖励和下一状态        │
        │     ↓                                                    │
        │  检测人工干预（Human-in-the-Loop）                        │
        │     ↓                                                    │
        │  构建 Transition (state, action, reward, next_state)      │
        │     ↓                                                    │
        │  Episode 结束时：                                         │
        │    - 发送 transitions 到 Learner                          │
        │    - 接收并更新策略参数                                   │
        │    - 发送统计数据（奖励、干预率等）                        │
        └─────────────────────────────────────────────────────────────┘

    Args:
        cfg: 训练配置对象
        shutdown_event: 关闭事件，用于检查是否应该停止执行
        parameters_queue: 从 Learner 接收更新参数的队列
        transitions_queue: 发送转换数据到 Learner 的队列
        interactions_queue: 发送统计数据到 Learner 的队列
    """
    # 多进程模式下，为子进程初始化独立的日志文件
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_policy_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor policy process logging initialized")

    logging.info("make_env online")

    # 创建在线环境和遥操作设备 ZCW
    online_env, teleop_device = make_robot_env(cfg=cfg.env)

    # 创建环境处理器和动作处理器
    # - env_processor: 处理观察数据（归一化、图像裁剪等）
    # - action_processor: 处理动作数据（缩放、模式转换等）
    env_processor, action_processor = make_processors(online_env, teleop_device, cfg.env, cfg.policy.device)

    # 设置随机种子以确保可重复性
    set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.policy.device, log=True)

    # 启用 CUDA 性能优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("make_policy")

    # 创建策略实例
    # 注意：Actor 和 Learner 都各自创建策略实例，通过传输参数而非整个对象来更新
    # 这样避免了序列化整个策略对象的开销ZCW
    policy: SACPolicy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )
    policy = policy.eval()  # 设置为评估模式（禁用 Dropout 等）
    assert isinstance(policy, nn.Module)

    # 重置环境，获取初始观察
    obs, info = online_env.reset()
    env_processor.reset()
    action_processor.reset()

    # 创建并处理初始转换
    transition = create_transition(observation=obs, info=info)
    transition = env_processor(transition)

    # 初始化 episode 统计变量
    # 注意：目前只处理单个环境的情况
    sum_reward_episode = 0
    list_transition_to_send_to_learner = []
    episode_intervention = False
    episode_intervention_steps = 0  # 人工干预步数计数
    episode_total_steps = 0         # 总步数计数

    # 策略推断计时器，用于监控 FPS
    policy_timer = TimerManager("Policy inference", log=False)

    # ============================================================================
    # 主循环：在环境中执行策略
    # ============================================================================
    for interaction_step in range(cfg.policy.online_steps):
        start_time = time.perf_counter()

        # 检查是否收到关闭信号
        if shutdown_event.is_set():
            logging.info("[ACTOR] Shutting down act_with_policy")
            return

        # 从当前转换中提取观察，只保留策略需要的特征
        observation = {
            k: v for k, v in transition[TransitionKey.OBSERVATION].items() if k in cfg.policy.input_features
        }

        # 策略推断：根据观察选择动作
        # 使用计时器测量 FPS，检查是否满足实时性要求
        with policy_timer:
            action = policy.select_action(batch=observation)
        policy_fps = policy_timer.fps_last

        # 如果 FPS 低于环境要求，记录警告
        log_policy_frequency_issue(policy_fps=policy_fps, cfg=cfg, interaction_step=interaction_step)

        # 执行动作并处理转换
        # 这个函数会：
        # 1. 将动作发送到环境
        # 2. 接收新的观察和奖励
        # 3. 检查是否有人工干预
        # 4. 处理新观察
        new_transition = step_env_and_process_transition(
            env=online_env,
            transition=transition,
            action=action,
            env_processor=env_processor,
            action_processor=action_processor,
        )

        # 提取处理后的下一状态观察
        next_observation = {
            k: v
            for k, v in new_transition[TransitionKey.OBSERVATION].items()
            if k in cfg.policy.input_features
        }

        # 获取实际执行的动作
        # 注意：这个动作可能是：
        # - 策略生成的动作（无人干预时）
        # - 人工遥操作的动作（有人干预时）
        executed_action = new_transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]

        # 提取奖励和终止标志
        reward = new_transition[TransitionKey.REWARD]
        done = new_transition.get(TransitionKey.DONE, False)
        truncated = new_transition.get(TransitionKey.TRUNCATED, False)

        # 累积 episode 奖励
        sum_reward_episode += float(reward)
        episode_total_steps += 1

        # 检查是否有人工干预
        intervention_info = new_transition[TransitionKey.INFO]
        if intervention_info.get(TeleopEvents.IS_INTERVENTION, False):
            episode_intervention = True
            episode_intervention_steps += 1

        # 构建辅助信息
        complementary_info = {
            "discrete_penalty": torch.tensor(
                [new_transition[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty", 0.0)]
            ),
        }

        # 创建转换对象并添加到待发送列表
        # 转换格式：(state, action, reward, next_state, done, truncated, complementary_info)
        list_transition_to_send_to_learner.append(
            Transition(
                state=observation,
                action=executed_action,
                reward=reward,
                next_state=next_observation,
                done=done,
                truncated=truncated,
                complementary_info=complementary_info,
            )
        )

        # 更新当前转换，为下一轮循环做准备
        transition = new_transition

        # ========================================================================
        # Episode 结束时的处理
        # ========================================================================
        if done or truncated:
            logging.info(f"[ACTOR] Global step {interaction_step}: Episode reward: {sum_reward_episode}")

            # 1. 更新策略参数（从 Learner 接收最新参数）
            update_policy_parameters(policy=policy, parameters_queue=parameters_queue, device=device)

            # 2. 发送转换数据到 Learner
            if len(list_transition_to_send_to_learner) > 0:
                push_transitions_to_transport_queue(
                    transitions=list_transition_to_send_to_learner,
                    transitions_queue=transitions_queue,
                )
                list_transition_to_send_to_learner = []

            # 3. 获取策略推断性能统计
            stats = get_frequency_stats(policy_timer)
            policy_timer.reset()

            # 4. 计算人工干预率
            intervention_rate = 0.0
            if episode_total_steps > 0:
                intervention_rate = episode_intervention_steps / episode_total_steps

            # 5. 发送 episode 统计数据到 Learner
            interactions_queue.put(
                python_object_to_bytes(
                    {
                        "Episodic reward": sum_reward_episode,
                        "Interaction step": interaction_step,
                        "Episode intervention": int(episode_intervention),
                        "Intervention rate": intervention_rate,
                        **stats,
                    }
                )
            )

            # 6. 重置 episode 计数器
            sum_reward_episode = 0.0
            episode_intervention = False
            episode_intervention_steps = 0
            episode_total_steps = 0

            # 7. 重置环境和处理器
            obs, info = online_env.reset()
            env_processor.reset()
            action_processor.reset()

            # 8. 处理初始观察
            transition = create_transition(observation=obs, info=info)
            transition = env_processor(transition)

        # 控制环境运行帧率
        if cfg.env.fps is not None:
            dt_time = time.perf_counter() - start_time
            precise_sleep(max(1 / cfg.env.fps - dt_time, 0.0))


# ============================================================================
# 通信函数 - gRPC 和消息传递相关函数
# ============================================================================

def establish_learner_connection(
    stub: services_pb2_grpc.LearnerServiceStub,
    shutdown_event: Event,  # type: ignore
    attempts: int = 30,
):
    """
    建立与 Learner 的连接

    通过发送 Ready 消息并等待响应来验证 Learner 是否可用。
    会尝试多次连接，每次失败后等待 2 秒。

    Args:
        stub: gRPC Learner 服务存根
        shutdown_event: 关闭事件，用于中断连接尝试
        attempts: 最大尝试次数（默认 30 次）

    Returns:
        bool: 连接成功返回 True，否则返回 False

    流程：
        ┌─────────────────┐
        │ 发送 Ready 请求  │
        └────────┬────────┘
                 │
         ┌───────┴───────┐
         │   收到响应？    │
         └───┬───────┬───┘
             │       │
            是       否
             │       │
             ▼       ▼
        ┌──────┐  ┌───────────┐
        │返回  │  │等待 2 秒   │
        │ True │  └─────┬─────┘
        └──────┘         │
                 ┌───────┴───────┐
                 │   尝试次数用尽？ │
                 └───┬───────┬───┘
                     │       │
                    否      是
                     │       │
                     ▼       ▼
                重试      返回 False
    """
    for _ in range(attempts):
        # 检查是否收到关闭信号
        if shutdown_event.is_set():
            logging.info("[ACTOR] Shutting down establish_learner_connection")
            return False

        # 尝试发送 Ready 消息来测试连接
        try:
            logging.info("[ACTOR] Send ready message to Learner")
            if stub.Ready(services_pb2.Empty()) == services_pb2.Empty():
                return True
        except grpc.RpcError as e:
            logging.error(f"[ACTOR] Waiting for Learner to be ready... {e}")
            time.sleep(2)
    return False


@lru_cache(maxsize=1)
def learner_service_client(
    host: str = "127.0.0.1",
    port: int = 50051,
) -> tuple[services_pb2_grpc.LearnerServiceStub, grpc.Channel]:
    """
    创建 Learner gRPC 客户端（带缓存）

    使用 @lru_cache 装饰器确保只创建一个客户端实例。
    gRPC 使用 HTTP/2 协议，可以在单个连接上多路复用请求，
    因此只需要创建一个客户端并重复使用。

    Args:
        host: Learner 服务器地址（默认 "127.0.0.1"）
        port: Learner 服务器端口（默认 50051）

    Returns:
        tuple: (gRPC 存根, gRPC 通道)

    Note:
        使用缓存是因为 gRPC 通道是重量级对象，不应该频繁创建。
        同一个通道可以在多个线程/进程间共享。
    """

    channel = grpc.insecure_channel(
        f"{host}:{port}",
        grpc_channel_options(),
    )
    stub = services_pb2_grpc.LearnerServiceStub(channel)
    logging.info("[ACTOR] Learner service client created")
    return stub, channel


def receive_policy(
    cfg: TrainRLServerPipelineConfig,
    parameters_queue: Queue,
    shutdown_event: Event,  # type: ignore
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
):
    """
    从 Learner 接收策略参数（运行在独立的通信线程/进程中）

    这个函数持续监听来自 Learner 的参数更新流：
    1. 建立 gRPC 连接并调用 StreamParameters RPC
    2. 接收序列化的参数数据（分块传输）
    3. 将参数放入队列供主线程更新策略

    Args:
        cfg: 训练配置对象
        parameters_queue: 用于接收参数的队列
        shutdown_event: 关闭事件
        learner_client: 可选的 gRPC 客户端（多进程模式下为 None）
        grpc_channel: 可选的 gRPC 通道（多进程模式下为 None）
    """
    logging.info("[ACTOR] Start receiving parameters from the Learner")

    # 多进程模式下，创建进程专属的日志文件
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_receive_policy_{os.getpid()}.log")

        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor receive policy process logging initialized")

        # 设置进程信号处理器
        _ = ProcessSignalHandler(use_threads=False, display_pid=True)

    # 多进程模式下，子进程需要创建自己的 gRPC 客户端
    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        # 调用 gRPC 流式 RPC，持续接收参数
        iterator = learner_client.StreamParameters(services_pb2.Empty())

        # 分块接收字节数据并放入队列
        receive_bytes_in_chunks(
            iterator,
            parameters_queue,
            shutdown_event,
            log_prefix="[ACTOR] parameters",
        )

    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    # 多进程模式下，关闭 gRPC 通道
    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Received policy loop stopped")


def send_transitions(
    cfg: TrainRLServerPipelineConfig,
    transitions_queue: Queue,
    shutdown_event: any,  # Event,
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
) -> services_pb2.Empty:
    """
    发送转换数据到 Learner（运行在独立的通信线程/进程中）

    这个函数持续从队列中获取转换数据并发送给 Learner：
    1. 从队列获取批次转换数据
    2. 通过 gRPC 流式 RPC 发送（支持大数据分块传输）

    Args:
        cfg: 训练配置对象
        transitions_queue: 包含待发送转换数据的队列
        shutdown_event: 关闭事件
        learner_client: 可选的 gRPC 客户端
        grpc_channel: 可选的 gRPC 通道

    Returns:
        services_pb2.Empty: gRPC 空响应
    """

    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_transitions_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor transitions process logging initialized")

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        learner_client.SendTransitions(
            transitions_stream(
                shutdown_event, transitions_queue, cfg.policy.actor_learner_config.queue_get_timeout
            )
        )
    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    logging.info("[ACTOR] Finished streaming transitions")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Transitions process stopped")


def send_interactions(
    cfg: TrainRLServerPipelineConfig,
    interactions_queue: Queue,
    shutdown_event: Event,  # type: ignore
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
) -> services_pb2.Empty:
    """
    发送交互统计数据到 Learner（运行在独立的通信线程/进程中）

    这个函数持续从队列中获取统计数据并发送给 Learner：
    1. 从队列获取 episode 统计信息
    2. 通过 gRPC 流式 RPC 发送

    统计数据包括：
    - Episode 奖励
    - 交互步数
    - 是否有人工干预
    - 干预率
    - 策略推断 FPS

    Args:
        cfg: 训练配置对象
        interactions_queue: 包含待发送统计数据的队列
        shutdown_event: 关闭事件
        learner_client: 可选的 gRPC 客户端
        grpc_channel: 可选的 gRPC 通道

    Returns:
        services_pb2.Empty: gRPC 空响应
    """

    # 多进程模式下，创建进程专属的日志文件
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_interactions_{os.getpid()}.log")

        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor interactions process logging initialized")

        # 设置进程信号处理器
        _ = ProcessSignalHandler(use_threads=False, display_pid=True)

    # 多进程模式下，创建自己的 gRPC 客户端
    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        learner_client.SendInteractions(
            interactions_stream(
                shutdown_event, interactions_queue, cfg.policy.actor_learner_config.queue_get_timeout
            )
        )
    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    logging.info("[ACTOR] Finished streaming interactions")

    # 多进程模式下，关闭 gRPC 通道
    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Interactions process stopped")


def transitions_stream(shutdown_event: Event, transitions_queue: Queue, timeout: float) -> services_pb2.Empty:  # type: ignore
    """
    生成器函数：从队列中获取转换数据并流式发送

    Args:
        shutdown_event: 关闭事件
        transitions_queue: 转换数据队列
        timeout: 队列获取超时时间

    Yields:
        services_pb2.Transition: gRPC 转换消息
    """
    while not shutdown_event.is_set():
        try:
            message = transitions_queue.get(block=True, timeout=timeout)
        except Empty:
            logging.debug("[ACTOR] Transition queue is empty")
            continue

        # 分块发送大数据
        yield from send_bytes_in_chunks(
            message, services_pb2.Transition, log_prefix="[ACTOR] Send transitions"
        )

    return services_pb2.Empty()


def interactions_stream(
    shutdown_event: Event,
    interactions_queue: Queue,
    timeout: float,  # type: ignore
) -> services_pb2.Empty:
    """
    生成器函数：从队列中获取统计数据并流式发送

    Args:
        shutdown_event: 关闭事件
        interactions_queue: 统计数据队列
        timeout: 队列获取超时时间

    Yields:
        services_pb2.InteractionMessage: gRPC 交互消息
    """
    while not shutdown_event.is_set():
        try:
            message = interactions_queue.get(block=True, timeout=timeout)
        except Empty:
            logging.debug("[ACTOR] Interaction queue is empty")
            continue

        # 分块发送数据
        yield from send_bytes_in_chunks(
            message,
            services_pb2.InteractionMessage,
            log_prefix="[ACTOR] Send interactions",
        )

    return services_pb2.Empty()


# ============================================================================
# 策略函数
# ============================================================================

def update_policy_parameters(policy: SACPolicy, parameters_queue: Queue, device):
    """
    更新 Actor 的策略参数

    从参数队列中获取最新的状态字典并加载到策略中。
    通常在 Episode 结束时调用。

    参数结构：
    {
        "policy": actor.state_dict(),           # Actor 网络参数
        "discrete_critic": discrete_critic.state_dict()  # 可选的离散批评家参数
    }

    Args:
        policy: 要更新的策略对象
        parameters_queue: 包含序列化参数的队列
        device: 目标设备（"cpu" 或 "cuda"）

    Note:
        TODO: 编码器参数同步问题
        1. 当 shared_encoder=True 时，应该发送 Critic 的编码器参数
        2. 当 freeze_vision_encoder=True 时，跳过编码器参数传输以节省带宽
    """
    # 非阻塞地获取最新参数
    bytes_state_dict = get_last_item_from_queue(parameters_queue, block=False)
    if bytes_state_dict is not None:
        logging.info("[ACTOR] Load new parameters from Learner.")
        state_dicts = bytes_to_state_dict(bytes_state_dict)

        # 加载 Actor 网络参数
        actor_state_dict = move_state_dict_to_device(state_dicts["policy"], device=device)
        policy.actor.load_state_dict(actor_state_dict)

        # 如果存在离散批评家，也加载其参数
        if hasattr(policy, "discrete_critic") and "discrete_critic" in state_dicts:
            discrete_critic_state_dict = move_state_dict_to_device(
                state_dicts["discrete_critic"], device=device
            )
            policy.discrete_critic.load_state_dict(discrete_critic_state_dict)
            logging.info("[ACTOR] Loaded discrete critic parameters from Learner.")


# ============================================================================
# 工具函数
# ============================================================================

def push_transitions_to_transport_queue(transitions: list, transitions_queue):
    """
    发送转换数据到传输队列

    将转换列表移动到 CPU，检查 NaN 值，然后序列化并发送到队列。

    Args:
        transitions: 待发送的转换列表
        transitions_queue: 传输队列

    处理步骤：
        1. 将转换数据移动到 CPU
        2. 检查是否存在 NaN 值
        3. 序列化为字节
        4. 放入传输队列
    """
    transition_to_send_to_learner = []
    for transition in transitions:
        # 将转换数据移动到 CPU（用于网络传输）
        tr = move_transition_to_device(transition=transition, device="cpu")

        # 检查观察数据中是否存在 NaN 值
        for key, value in tr["state"].items():
            if torch.isnan(value).any():
                logging.warning(f"Found NaN values in transition {key}")

        transition_to_send_to_learner.append(tr)

    # 序列化并发送到队列
    transitions_queue.put(transitions_to_bytes(transition_to_send_to_learner))


def get_frequency_stats(timer: TimerManager) -> dict[str, float]:
    """
    获取策略推断的频率统计信息

    Args:
        timer: 包含性能指标的计时器

    Returns:
        dict[str, float]: 频率统计信息，包括：
            - "Policy frequency [Hz]": 平均 FPS
            - "Policy frequency 90th-p [Hz]": 90分位 FPS
    """
    stats = {}
    if timer.count > 1:
        avg_fps = timer.fps_avg
        p90_fps = timer.fps_percentile(90)
        logging.debug(f"[ACTOR] Average policy frame rate: {avg_fps}")
        logging.debug(f"[ACTOR] Policy frame rate 90th percentile: {p90_fps}")
        stats = {
            "Policy frequency [Hz]": avg_fps,
            "Policy frequency 90th-p [Hz]": p90_fps,
        }
    return stats


def log_policy_frequency_issue(policy_fps: float, cfg: TrainRLServerPipelineConfig, interaction_step: int):
    """
    记录策略 FPS 问题警告

    当策略推断 FPS 低于环境要求的 FPS 时记录警告。

    Args:
        policy_fps: 当前策略推断 FPS
        cfg: 训练配置对象
        interaction_step: 当前交互步数
    """
    if policy_fps < cfg.env.fps:
        logging.warning(
            f"[ACTOR] Policy FPS {policy_fps:.1f} below required {cfg.env.fps} at step {interaction_step}"
        )


def use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    """
    判断是否使用线程模式

    Args:
        cfg: 训练配置对象

    Returns:
        bool: True 表示使用线程，False 表示使用进程
    """
    return cfg.policy.concurrency.actor == "threads"


# ============================================================================
# 程序入口点
# ============================================================================

if __name__ == "__main__":
    actor_cli()
