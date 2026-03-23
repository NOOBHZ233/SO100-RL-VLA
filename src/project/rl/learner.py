# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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
Learner server runner for distributed HILSerl robot policy training.

This script implements the learner component of the distributed HILSerl architecture.
It initializes the policy network, maintains replay buffers, and updates
the policy based on transitions received from the actor server.

Examples of usage:

- Start a learner server for training:
```bash
python -m lerobot.rl.learner --config_path src/lerobot/configs/train_config_hilserl_so100.json
```

**NOTE**: Start the learner server before launching the actor server. The learner opens a gRPC server
to communicate with actors.

**NOTE**: Training progress can be monitored through Weights & Biases if wandb.enable is set to true
in your configuration.

**WORKFLOW**:
1. Create training configuration with proper policy, dataset, and environment settings
2. Start this learner server with the configuration
3. Start an actor server with the same configuration
4. Monitor training progress through wandb dashboard

For more details on the complete HILSerl training workflow, see:
https://github.com/michel-aractingi/lerobot-hilserl-guide
"""

import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pprint import pformat

import grpc
import torch
from termcolor import colored
from torch import nn
from torch.multiprocessing import Queue
from torch.optim.optimizer import Optimizer

from project.cameras import opencv  # noqa: F401
from project.configs import parser
from project.configs.train import TrainRLServerPipelineConfig
from project.datasets.factory import make_dataset
from project.datasets.lerobot_dataset import LeRobotDataset
from project.policies.factory import make_policy
from project.policies.sac.modeling_sac import SACPolicy
from project.rl.buffer import ReplayBuffer, concatenate_batch_transitions
from project.rl.process import ProcessSignalHandler
from project.rl.wandb_utils import WandBLogger
from project.robots import so100_follower  # noqa: F401
from project.teleoperators.utils import TeleopEvents
from project.transport import services_pb2_grpc
from project.transport.utils import (
    MAX_MESSAGE_SIZE,
    bytes_to_python_object,
    bytes_to_transitions,
    state_to_bytes,
)
from project.utils.constans import (
    ACTION,
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
)
from project.utils.random_utils import set_seed
from project.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state as utils_load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from project.utils.transition import move_state_dict_to_device, move_transition_to_device
from project.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
)

from .learner_service import MAX_WORKERS, SHUTDOWN_TIMEOUT, LearnerService


@parser.wrap()
def train_cli(cfg: TrainRLServerPipelineConfig):
    """
    命令行接口入口函数，用于启动RL训练过程。

    根据配置决定使用多进程还是多线程模式，并调用训练主函数。

    Args:
        cfg (TrainRLServerPipelineConfig): 训练配置对象，包含所有训练相关的参数
    """
    # 如果不使用线程模式，则使用多进程模式
    # 设置spawn启动方式以确保在不同平台上的兼容性
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")

    # 使用配置中的job_name启动训练
    train(
        cfg,
        job_name=cfg.job_name,
    )

    logging.info("[LEARNER] train_cli finished")


def train(cfg: TrainRLServerPipelineConfig, job_name: str | None = None):
    """
    Main training function that initializes and runs the training process.

    Args:
        cfg (TrainRLServerPipelineConfig): The training configuration
        job_name (str | None, optional): Job name for logging. Defaults to None.
    """

    cfg.validate()

    if job_name is None:
        job_name = cfg.job_name

    if job_name is None:
        raise ValueError("Job name must be specified either in config or as a parameter")

    display_pid = False
    if not use_threads(cfg):
        display_pid = True

    # Create logs directory to ensure it exists
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"learner_{job_name}.log")

    # Initialize logging with explicit log file
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Learner logging initialized, writing to {log_file}")
    try:
        logging.info(pformat(cfg.to_dict()))
    except TypeError:
        logging.warning("Failed to encode config to dict due to typing.Any. Logging as string instead.")
        logging.info(str(cfg))

    # Setup WandB logging if enabled
    if cfg.wandb.enable and cfg.wandb.project:
        from project.rl.wandb_utils import WandBLogger

        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    # Handle resume logic
    cfg = handle_resume_logic(cfg)

    set_seed(seed=cfg.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    start_learner_threads(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
    )


def start_learner_threads(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,  # Event,
) -> None:
    """
    Start the learner threads for training.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        wandb_logger (WandBLogger | None): Logger for metrics
        shutdown_event: Event to signal shutdown
    """
    # Create multiprocessing queues
    transition_queue = Queue()
    interaction_message_queue = Queue()
    parameters_queue = Queue()

    concurrency_entity = None

    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from torch.multiprocessing import Process

        concurrency_entity = Process

    communication_process = concurrency_entity(
        target=start_learner,
        args=(
            parameters_queue,
            transition_queue,
            interaction_message_queue,
            shutdown_event,
            cfg,
        ),
        daemon=True,
    )
    communication_process.start()

    add_actor_information_and_train(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        parameters_queue=parameters_queue,
    )
    logging.info("[LEARNER] Training process stopped")

    logging.info("[LEARNER] Closing queues")
    transition_queue.close()
    interaction_message_queue.close()
    parameters_queue.close()

    communication_process.join()
    logging.info("[LEARNER] Communication process joined")

    logging.info("[LEARNER] join queues")
    transition_queue.cancel_join_thread()
    interaction_message_queue.cancel_join_thread()
    parameters_queue.cancel_join_thread()

    logging.info("[LEARNER] queues closed")


# Core algorithm functions


def add_actor_information_and_train(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,  # Event,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    parameters_queue: Queue,
):
    """
    Handles data transfer from the actor to the learner, manages training updates,
    and logs training progress in an online reinforcement learning setup.

    This function continuously:
    - Transfers transitions from the actor to the replay buffer.
    - Logs received interaction messages.
    - Ensures training begins only when the replay buffer has a sufficient number of transitions.
    - Samples batches from the replay buffer and performs multiple critic updates.
    - Periodically updates the actor, critic, and temperature optimizers.
    - Logs training statistics, including loss values and optimization frequency.

    NOTE: This function doesn't have a single responsibility, it should be split into multiple functions
    in the future. The reason why we did that is the  GIL in Python. It's super slow the performance
    are divided by 200. So we need to have a single thread that does all the work.

    Args:
        cfg (TrainRLServerPipelineConfig): Configuration object containing hyperparameters.
        wandb_logger (WandBLogger | None): Logger for tracking training progress.
        shutdown_event (Event): Event to signal shutdown.
        transition_queue (Queue): Queue for receiving transitions from the actor.
        interaction_message_queue (Queue): Queue for receiving interaction messages from the actor.
        parameters_queue (Queue): Queue for sending policy parameters to the actor.
    """
    # 在函数开始处提取所有配置变量，这样做可以将性能提升约7%
    # 避免在热循环中重复访问配置对象的属性
    device = get_safe_torch_device(try_device=cfg.policy.device, log=True)
    storage_device = get_safe_torch_device(try_device=cfg.policy.storage_device)
    clip_grad_norm_value = cfg.policy.grad_clip_norm  # 梯度裁剪阈值，防止梯度爆炸
    online_step_before_learning = cfg.policy.online_step_before_learning  # 开始学习前需要的最少步数
    utd_ratio = cfg.policy.utd_ratio  # Update-To-Data ratio，每个数据点更新的次数
    fps = cfg.env.fps  # 环境帧率
    log_freq = cfg.log_freq  # 日志记录频率
    save_freq = cfg.save_freq  # 检查点保存频率
    policy_update_freq = cfg.policy.policy_update_freq  # Actor策略更新频率
    policy_parameters_push_frequency = cfg.policy.actor_learner_config.policy_parameters_push_frequency
    saving_checkpoint = cfg.save_checkpoint  # 是否保存检查点
    online_steps = cfg.policy.online_steps  # 在线训练总步数
    async_prefetch = cfg.policy.async_prefetch  # 是否异步预取数据

    # 初始化多进程环境的日志记录
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_train_process_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Initialized logging for actor information and training process")

    logging.info("Initializing policy")

    # 创建策略网络（SAC算法的Actor-Critic架构）
    policy: SACPolicy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )

    assert isinstance(policy, nn.Module)

    # 设置为训练模式（启用dropout、batch norm等训练特定行为）
    policy.train()

    # 将初始策略参数推送到队列，供Actor获取
    push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)

    last_time_policy_pushed = time.time()

    # 创建Actor、Critic和温度参数的优化器
    optimizers, lr_scheduler = make_optimizers_and_scheduler(cfg=cfg, policy=policy)

    # 如果是从检查点恢复训练，加载训练状态（优化器状态、步数等）
    resume_optimization_step, resume_interaction_step = load_training_state(cfg=cfg, optimizers=optimizers)

    # 记录训练信息（参数量、输出目录等）
    log_training_info(cfg=cfg, policy=policy)

    # 初始化在线经验回放缓冲区（存储与环境交互产生的经验）
    replay_buffer = initialize_replay_buffer(cfg, device, storage_device)
    batch_size = cfg.batch_size
    offline_replay_buffer = None

    # 如果配置了离线数据集，初始化离线经验回放缓冲区
    # 这用于离线强化学习（offline RL）或从演示数据中预训练
    if cfg.dataset is not None:
        offline_replay_buffer = initialize_offline_replay_buffer(
            cfg=cfg,
            device=device,
            storage_device=storage_device,
        )
        # 同时从在线和离线缓冲区采样，因此各取一半批次大小
        batch_size: int = batch_size // 2  # We will sample from both replay buffer

    logging.info("Starting learner thread")
    interaction_message = None
    optimization_step = resume_optimization_step if resume_optimization_step is not None else 0
    interaction_step_shift = resume_interaction_step if resume_interaction_step is not None else 0

    dataset_repo_id = None
    if cfg.dataset is not None:
        dataset_repo_id = cfg.dataset.repo_id

    # 初始化数据迭代器（用于高效地从缓冲区批量采样）
    online_iterator = None
    offline_iterator = None

    # NOTE: 这是Learner的主循环
    while True:
        # 如果收到关闭信号，退出训练循环
        if shutdown_event is not None and shutdown_event.is_set():
            logging.info("[LEARNER] Shutdown signal received. Exiting...")
            break

        # 处理所有从Actor服务器发送过来的转换数据，存入回放缓冲区
        process_transitions(
            transition_queue=transition_queue,
            replay_buffer=replay_buffer,
            offline_replay_buffer=offline_replay_buffer,
            device=device,
            dataset_repo_id=dataset_repo_id,
            shutdown_event=shutdown_event,
        )

        # 处理所有从Actor服务器发送过来的交互消息
        interaction_message = process_interaction_messages(
            interaction_message_queue=interaction_message_queue,
            interaction_step_shift=interaction_step_shift,
            wandb_logger=wandb_logger,
            shutdown_event=shutdown_event,
        )

        # 等待回放缓冲区积累足够的样本后再开始训练
        # 这是RL训练的常见做法，避免初始训练不稳定
        if len(replay_buffer) < online_step_before_learning:
            continue

        # 延迟初始化迭代器，确保缓冲区有足够数据后才创建
        if online_iterator is None:
            online_iterator = replay_buffer.get_iterator(
                batch_size=batch_size, async_prefetch=async_prefetch, queue_size=2
            )

        if offline_replay_buffer is not None and offline_iterator is None:
            offline_iterator = offline_replay_buffer.get_iterator(
                batch_size=batch_size, async_prefetch=async_prefetch, queue_size=2
            )

        time_for_one_optimization_step = time.time()
        # UTD (Update-To-Data) ratio: 每个数据样本进行多次梯度更新
        # 这可以提高样本效率，是现代RL算法的常用技术
        for _ in range(utd_ratio - 1):
            # 从迭代器中采样一个批次
            batch = next(online_iterator)

            # 如果有离线数据集，同时从离线缓冲区采样并合并
            if dataset_repo_id is not None:
                batch_offline = next(offline_iterator)
                batch = concatenate_batch_transitions(
                    left_batch_transitions=batch, right_batch_transition=batch_offline
                )

            # 解包批次数据
            actions = batch[ACTION]
            rewards = batch["reward"]
            observations = batch["state"]
            next_observations = batch["next_state"]
            done = batch["done"]
            # 检查数据中是否存在NaN值（可能导致训练不稳定）
            check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)

            # 获取观察特征（如果视觉编码器冻结，使用缓存的特征以节省计算）
            observation_features, next_observation_features = get_observation_features(
                policy=policy, observations=observations, next_observations=next_observations
            )

            # 创建forward方法所需的批次字典
            forward_batch = {
                ACTION: actions,
                "reward": rewards,
                "state": observations,
                "next_state": next_observations,
                "done": done,
                "observation_feature": observation_features,
                "next_observation_feature": next_observation_features,
                "complementary_info": batch["complementary_info"],
            }

            # 使用Critic模型计算损失（评估当前状态-动作对的价值）
            critic_output = policy.forward(forward_batch, model="critic")

            # 执行Critic优化：计算梯度、裁剪梯度、更新参数
            loss_critic = critic_output["loss_critic"]
            optimizers["critic"].zero_grad()
            loss_critic.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
            )
            optimizers["critic"].step()

            # 离散动作Critic优化（如果可用）
            if policy.config.num_discrete_actions is not None:
                discrete_critic_output = policy.forward(forward_batch, model="discrete_critic")
                loss_discrete_critic = discrete_critic_output["loss_discrete_critic"]
                optimizers["discrete_critic"].zero_grad()
                loss_discrete_critic.backward()
                discrete_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value
                )
                optimizers["discrete_critic"].step()

            # 更新目标网络（使用软更新策略）
            policy.update_target_networks()

        # UTD比率的最后一次更新：这次更新会同时进行Actor优化
        batch = next(online_iterator)

        if dataset_repo_id is not None:
            batch_offline = next(offline_iterator)
            batch = concatenate_batch_transitions(
                left_batch_transitions=batch, right_batch_transition=batch_offline
            )

        actions = batch[ACTION]
        rewards = batch["reward"]
        observations = batch["state"]
        next_observations = batch["next_state"]
        done = batch["done"]

        check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)

        observation_features, next_observation_features = get_observation_features(
            policy=policy, observations=observations, next_observations=next_observations
        )

        # 创建forward方法所需的批次字典
        forward_batch = {
            ACTION: actions,
            "reward": rewards,
            "state": observations,
            "next_state": next_observations,
            "done": done,
            "observation_feature": observation_features,
            "next_observation_feature": next_observation_features,
        }

        # 计算Critic损失
        critic_output = policy.forward(forward_batch, model="critic")

        # 执行Critic优化
        loss_critic = critic_output["loss_critic"]
        optimizers["critic"].zero_grad()
        loss_critic.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
        ).item()
        optimizers["critic"].step()

        # 初始化训练信息字典，用于记录和日志
        training_infos = {
            "loss_critic": loss_critic.item(),
            "critic_grad_norm": critic_grad_norm,
        }

        # 离散动作Critic优化（如果可用）
        if policy.config.num_discrete_actions is not None:
            discrete_critic_output = policy.forward(forward_batch, model="discrete_critic")
            loss_discrete_critic = discrete_critic_output["loss_discrete_critic"]
            optimizers["discrete_critic"].zero_grad()
            loss_discrete_critic.backward()
            discrete_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value
            ).item()
            optimizers["discrete_critic"].step()

            # 将离散Critic信息添加到训练信息中
            training_infos["loss_discrete_critic"] = loss_discrete_critic.item()
            training_infos["discrete_critic_grad_norm"] = discrete_critic_grad_norm

        # Actor和温度参数优化（按指定频率执行）
        # Actor不需要每次都更新，可以降低频率以提高训练效率
        if optimization_step % policy_update_freq == 0:
            for _ in range(policy_update_freq):
                # Actor优化：更新策略网络以最大化期望回报
                actor_output = policy.forward(forward_batch, model="actor")
                loss_actor = actor_output["loss_actor"]
                optimizers["actor"].zero_grad()
                loss_actor.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.actor.parameters(), max_norm=clip_grad_norm_value
                ).item()
                optimizers["actor"].step()

                # 将Actor信息添加到训练信息中
                training_infos["loss_actor"] = loss_actor.item()
                training_infos["actor_grad_norm"] = actor_grad_norm

                # 温度参数优化：控制策略的熵（探索程度）
                # 在SAC算法中，温度参数自动调节以维持目标熵值
                temperature_output = policy.forward(forward_batch, model="temperature")
                loss_temperature = temperature_output["loss_temperature"]
                optimizers["temperature"].zero_grad()
                loss_temperature.backward()
                temp_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=[policy.log_alpha], max_norm=clip_grad_norm_value
                ).item()
                optimizers["temperature"].step()

                # 将温度参数信息添加到训练信息中
                training_infos["loss_temperature"] = loss_temperature.item()
                training_infos["temperature_grad_norm"] = temp_grad_norm
                training_infos["temperature"] = policy.temperature

        # 根据配置的频率将策略推送到Actor
        # 这样Actor可以定期获取最新的策略参数
        if time.time() - last_time_policy_pushed > policy_parameters_push_frequency:
            push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)
            last_time_policy_pushed = time.time()

        # 更新目标网络（使用软更新策略）
        policy.update_target_networks()

        # 按指定间隔记录训练指标
        if optimization_step % log_freq == 0:
            training_infos["replay_buffer_size"] = len(replay_buffer)
            if offline_replay_buffer is not None:
                training_infos["offline_replay_buffer_size"] = len(offline_replay_buffer)
            training_infos["Optimization step"] = optimization_step

            # 记录训练指标到WandB（如果启用）
            if wandb_logger:
                wandb_logger.log_dict(d=training_infos, mode="train", custom_step_key="Optimization step")

        # 计算并记录优化频率（用于性能监控）
        time_for_one_optimization_step = time.time() - time_for_one_optimization_step
        frequency_for_one_optimization_step = 1 / (time_for_one_optimization_step + 1e-9)

        logging.info(f"[LEARNER] Optimization frequency loop [Hz]: {frequency_for_one_optimization_step}")

        # 记录优化频率到WandB
        if wandb_logger:
            wandb_logger.log_dict(
                {
                    "Optimization frequency loop [Hz]": frequency_for_one_optimization_step,
                    "Optimization step": optimization_step,
                },
                mode="train",
                custom_step_key="Optimization step",
            )

        optimization_step += 1
        if optimization_step % log_freq == 0:
            logging.info(f"[LEARNER] Number of optimization step: {optimization_step}")

        # 按指定间隔保存检查点
        if saving_checkpoint and (optimization_step % save_freq == 0 or optimization_step == online_steps):
            save_training_checkpoint(
                cfg=cfg,
                optimization_step=optimization_step,
                online_steps=online_steps,
                interaction_message=interaction_message,
                policy=policy,
                optimizers=optimizers,
                replay_buffer=replay_buffer,
                offline_replay_buffer=offline_replay_buffer,
                dataset_repo_id=dataset_repo_id,
                fps=fps,
            )


def start_learner(
    parameters_queue: Queue,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    shutdown_event: any,  # Event,
    cfg: TrainRLServerPipelineConfig,
):
    """
    启动Learner服务器用于训练。

    该函数启动一个gRPC服务器，用于与Actor服务器进行通信：
    - 接收从Actor发送的转换数据和交互消息
    - 向Actor发送策略参数更新

    Args:
        parameters_queue (Queue): 用于向Actor发送策略参数的队列
        transition_queue (Queue): 用于接收Actor发送的转换数据的队列
        interaction_message_queue (Queue): 用于接收Actor发送的交互消息的队列
        shutdown_event (Event): 用于通知关闭的事件对象
        cfg (TrainRLServerPipelineConfig): 训练配置
    """
    # 如果使用多进程模式，创建进程专用的日志文件
    if not use_threads(cfg):
        # 创建进程专用的日志文件
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_process_{os.getpid()}.log")

        # 使用显式日志文件初始化日志
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Learner server process logging initialized")

        # 设置进程处理器来处理关闭信号
        # 但使用主进程的关闭事件
        # TODO: 检查是否有用
        _ = ProcessSignalHandler(False, display_pid=True)

    # 创建Learner服务，处理与Actor的gRPC通信
    service = LearnerService(
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        seconds_between_pushes=cfg.policy.actor_learner_config.policy_parameters_push_frequency,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        queue_get_timeout=cfg.policy.actor_learner_config.queue_get_timeout,
    )

    # 创建gRPC服务器，配置线程池和消息大小限制
    server = grpc.server(
        ThreadPoolExecutor(max_workers=MAX_WORKERS),
        options=[
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
        ],
    )

    # 将Learner服务添加到gRPC服务器
    services_pb2_grpc.add_LearnerServiceServicer_to_server(
        service,
        server,
    )

    # 获取服务器地址和端口
    host = cfg.policy.actor_learner_config.learner_host
    port = cfg.policy.actor_learner_config.learner_port

    # 绑定端口并启动服务器（使用不安全的连接）
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logging.info("[LEARNER] gRPC server started")

    # 等待关闭信号
    shutdown_event.wait()
    logging.info("[LEARNER] Stopping gRPC server...")
    server.stop(SHUTDOWN_TIMEOUT)
    logging.info("[LEARNER] gRPC server stopped")


def save_training_checkpoint(
    cfg: TrainRLServerPipelineConfig,
    optimization_step: int,
    online_steps: int,
    interaction_message: dict | None,
    policy: nn.Module,
    optimizers: dict[str, Optimizer],
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer | None = None,
    dataset_repo_id: str | None = None,
    fps: int = 30,
) -> None:
    """
    保存训练检查点和相关数据。

    该函数执行以下步骤：
    1. 使用当前优化步数创建检查点目录
    2. 保存策略模型、配置和优化器状态
    3. 保存当前交互步数用于恢复训练
    4. 更新"last"符号链接指向此检查点
    5. 将回放缓冲区保存为数据集以供后续使用
    6. 如果存在离线回放缓冲区，将其保存为单独的数据集

    Args:
        cfg: 训练配置
        optimization_step: 当前优化步数
        online_steps: 在线训练总步数
        interaction_message: 包含交互信息的字典
        policy: 要保存的策略模型
        optimizers: 优化器字典
        replay_buffer: 要保存为数据集的回放缓冲区
        offline_replay_buffer: 可选的离线回放缓冲区
        dataset_repo_id: 数据集仓库ID
        fps: 数据集的帧率
    """
    logging.info(f"Checkpoint policy after step {optimization_step}")
    _num_digits = max(6, len(str(online_steps)))
    interaction_step = interaction_message["Interaction step"] if interaction_message is not None else 0

    # 创建检查点目录
    checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, online_steps, optimization_step)

    # 保存检查点（模型、配置、优化器状态）
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=optimization_step,
        cfg=cfg,
        policy=policy,
        optimizer=optimizers,
        scheduler=None,
    )

    # 手动保存交互步数（用于恢复训练时的步数对齐）
    training_state_dir = os.path.join(checkpoint_dir, TRAINING_STATE_DIR)
    os.makedirs(training_state_dir, exist_ok=True)
    training_state = {"step": optimization_step, "interaction_step": interaction_step}
    torch.save(training_state, os.path.join(training_state_dir, "training_state.pt"))

    # 更新"last"符号链接指向最新检查点
    update_last_checkpoint(checkpoint_dir)

    # TODO: 临时在此保存回放缓冲区，在机器人上部署时可以移除
    # 我们希望通过键盘输入来控制此操作
    dataset_dir = os.path.join(cfg.output_dir, "dataset")
    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)

    # 保存数据集
    # 注意：处理配置中未指定数据集仓库ID的情况
    # 例如：没有演示数据的RL训练
    repo_id_buffer_save = cfg.env.task if dataset_repo_id is None else dataset_repo_id
    replay_buffer.to_lerobot_dataset(repo_id=repo_id_buffer_save, fps=fps, root=dataset_dir)

    # 如果存在离线回放缓冲区，也保存为数据集
    if offline_replay_buffer is not None:
        dataset_offline_dir = os.path.join(cfg.output_dir, "dataset_offline")
        if os.path.exists(dataset_offline_dir) and os.path.isdir(dataset_offline_dir):
            shutil.rmtree(dataset_offline_dir)

        offline_replay_buffer.to_lerobot_dataset(
            cfg.dataset.repo_id,
            fps=fps,
            root=dataset_offline_dir,
        )

    logging.info("Resume training")


def make_optimizers_and_scheduler(cfg: TrainRLServerPipelineConfig, policy: nn.Module):
    """
    为强化学习策略的Actor、Critic和温度组件创建优化器。

    该函数为以下组件设置Adam优化器：
    - **Actor网络**：确保只优化相关参数（如果编码器共享则排除编码器参数）
    - **Critic集成**：评估值函数
    - **温度参数**：控制SAC类方法中的熵

    同时初始化学习率调度器（目前设置为None）。

    注意事项：
    - 如果编码器是共享的，其参数将从Actor的优化过程中排除
    - 策略的log温度（log_alpha）被包装在列表中以确保作为独立张量进行优化

    Args:
        cfg (TrainRLServerPipelineConfig): 包含超参数的配置对象
        policy (nn.Module): 包含Actor、Critic和温度组件的策略模型

    Returns:
        tuple: 包含以下内容的元组：
            - optimizers (Dict[str, torch.optim.Optimizer]): 将组件名称映射到其Adam优化器的字典
            - lr_scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): 目前为None，可扩展支持学习率调度
    """
    # 创建Actor优化器
    # 如果编码器是共享的，排除编码器参数（它们将由Critic优化器更新）
    optimizer_actor = torch.optim.Adam(
        params=[
            p
            for n, p in policy.actor.named_parameters()
            if not policy.config.shared_encoder or not n.startswith("encoder")
        ],
        lr=cfg.policy.actor_lr,
    )
    # 创建Critic优化器（优化Critic集成网络）
    optimizer_critic = torch.optim.Adam(params=policy.critic_ensemble.parameters(), lr=cfg.policy.critic_lr)

    # 如果存在离散动作空间，创建离散Critic优化器
    if cfg.policy.num_discrete_actions is not None:
        optimizer_discrete_critic = torch.optim.Adam(
            params=policy.discrete_critic.parameters(), lr=cfg.policy.critic_lr
        )
    # 创建温度参数优化器（优化log_alpha参数）
    optimizer_temperature = torch.optim.Adam(params=[policy.log_alpha], lr=cfg.policy.critic_lr)
    lr_scheduler = None
    # 构建优化器字典
    optimizers = {
        "actor": optimizer_actor,
        "critic": optimizer_critic,
        "temperature": optimizer_temperature,
    }
    # 如果存在离散Critic，添加到优化器字典
    if cfg.policy.num_discrete_actions is not None:
        optimizers["discrete_critic"] = optimizer_discrete_critic
    return optimizers, lr_scheduler


# Training setup functions


def handle_resume_logic(cfg: TrainRLServerPipelineConfig) -> TrainRLServerPipelineConfig:
    """
    处理训练的恢复逻辑。

    如果resume为True：
    - 验证检查点是否存在
    - 加载检查点配置
    - 记录恢复详情
    - 返回检查点配置

    如果resume为False：
    - 检查输出目录是否存在（防止意外覆盖）
    - 返回原始配置

    Args:
        cfg (TrainRLServerPipelineConfig): 训练配置

    Returns:
        TrainRLServerPipelineConfig: 更新后的配置

    Raises:
        RuntimeError: 如果resume为True但未找到检查点，或resume为False但目录已存在
    """
    out_dir = cfg.output_dir

    # 情况1：不恢复训练，但需要检查目录是否存在以防止覆盖
    if not cfg.resume:
        checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
        if os.path.exists(checkpoint_dir):
            raise RuntimeError(
                f"Output directory {checkpoint_dir} already exists. Use `resume=true` to resume training."
            )
        return cfg

    # 情况2：恢复训练
    checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
    if not os.path.exists(checkpoint_dir):
        raise RuntimeError(f"No model checkpoint found in {checkpoint_dir} for resume=True")

    # 记录找到有效检查点并正在恢复
    logging.info(
        colored(
            "Valid checkpoint found: resume=True detected, resuming previous run",
            color="yellow",
            attrs=["bold"],
        )
    )

    # 使用Draccus加载配置
    checkpoint_cfg_path = os.path.join(checkpoint_dir, PRETRAINED_MODEL_DIR, "train_config.json")
    checkpoint_cfg = TrainRLServerPipelineConfig.from_pretrained(checkpoint_cfg_path)

    # 确保返回的配置中设置了resume标志
    checkpoint_cfg.resume = True
    return checkpoint_cfg


def load_training_state(
    cfg: TrainRLServerPipelineConfig,
    optimizers: Optimizer | dict[str, Optimizer],
):
    """
    从检查点加载训练状态（优化器、步数等）。

    Args:
        cfg (TrainRLServerPipelineConfig): 训练配置
        optimizers (Optimizer | dict): 要加载状态的优化器

    Returns:
        tuple: (optimization_step, interaction_step) 或 (None, None) 如果不恢复训练
    """
    if not cfg.resume:
        return None, None

    # 构建到最后检查点目录的路径
    checkpoint_dir = os.path.join(cfg.output_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)

    logging.info(f"Loading training state from {checkpoint_dir}")

    try:
        # 使用train_utils中的工具函数加载优化器状态
        step, optimizers, _ = utils_load_training_state(Path(checkpoint_dir), optimizers, None)

        # 从training_state.pt单独加载交互步数
        training_state_path = os.path.join(checkpoint_dir, TRAINING_STATE_DIR, "training_state.pt")
        interaction_step = 0
        if os.path.exists(training_state_path):
            # nosec B614: Safe usage of torch.load with trusted checkpoint
            training_state = torch.load(training_state_path, weights_only=False)
            interaction_step = training_state.get("interaction_step", 0)

        logging.info(f"Resuming from step {step}, interaction step {interaction_step}")
        return step, interaction_step

    except Exception as e:
        logging.error(f"Failed to load training state: {e}")
        return None, None


def log_training_info(cfg: TrainRLServerPipelineConfig, policy: nn.Module) -> None:
    """
    Log information about the training process.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        policy (nn.Module): Policy model
    """
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.policy.online_steps=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")


def initialize_replay_buffer(
    cfg: TrainRLServerPipelineConfig, device: str, storage_device: str
) -> ReplayBuffer:
    """
    初始化回放缓冲区，可以是空的或从数据集加载（如果是恢复训练）。

    Args:
        cfg (TrainRLServerPipelineConfig): 训练配置
        device (str): 存储张量的设备
        storage_device (str): 存储优化的设备

    Returns:
        ReplayBuffer: 初始化的回放缓冲区
    """
    if not cfg.resume:
        # 如果不是恢复训练，创建一个空的回放缓冲区
        return ReplayBuffer(
            capacity=cfg.policy.online_buffer_capacity,
            device=device,
            state_keys=cfg.policy.input_features.keys(),
            storage_device=storage_device,
            optimize_memory=True,
        )

    # 如果是恢复训练，从保存的数据集加载回放缓冲区
    logging.info("Resume training load the online dataset")
    dataset_path = os.path.join(cfg.output_dir, "dataset")

    # 注意：在RL中可能没有数据集（纯在线学习）
    repo_id = None
    if cfg.dataset is not None:
        repo_id = cfg.dataset.repo_id
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_path,
    )
    # 从LeRobot数据集创建回放缓冲区
    return ReplayBuffer.from_lerobot_dataset(
        lerobot_dataset=dataset,
        capacity=cfg.policy.online_buffer_capacity,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        optimize_memory=True,
    )


def initialize_offline_replay_buffer(
    cfg: TrainRLServerPipelineConfig,
    device: str,
    storage_device: str,
) -> ReplayBuffer:
    """
    从数据集初始化离线回放缓冲区。

    该函数用于离线强化学习或从演示数据中预训练。
    如果是恢复训练，从保存的数据集加载；否则创建新数据集。

    Args:
        cfg (TrainRLServerPipelineConfig): 训练配置
        device (str): 存储张量的设备
        storage_device (str): 存储优化的设备

    Returns:
        ReplayBuffer: 初始化的离线回放缓冲区
    """
    if not cfg.resume:
        # 如果不是恢复训练，从配置创建数据集
        logging.info("make_dataset offline buffer")
        offline_dataset = make_dataset(cfg)
    else:
        # 如果是恢复训练，从保存的数据集加载
        logging.info("load offline dataset")
        dataset_offline_path = os.path.join(cfg.output_dir, "dataset_offline")
        offline_dataset = LeRobotDataset(
            repo_id=cfg.dataset.repo_id,
            root=dataset_offline_path,
        )

    # 将LeRobot数据集转换为离线回放缓冲区
    logging.info("Convert to a offline replay buffer")
    offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
        offline_dataset,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        storage_device=storage_device,
        optimize_memory=True,
        capacity=cfg.policy.offline_buffer_capacity,
    )
    return offline_replay_buffer


# Utilities/Helpers functions


def get_observation_features(
    policy: SACPolicy, observations: torch.Tensor, next_observations: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    从策略编码器获取观察特征。该函数作为观察特征的缓存。

    当编码器被冻结时，观察特征不会更新。
    我们可以通过缓存观察特征来节省计算开销。

    Args:
        policy (SACPolicy): 策略模型
        observations (torch.Tensor): 当前观察
        next_observations (torch.Tensor): 下一步观察

    Returns:
        tuple: (observation_features, next_observation_features) 如果编码器冻结，否则返回 (None, None)
    """
    # 如果没有视觉编码器或编码器未被冻结，返回None
    if policy.config.vision_encoder_name is None or not policy.config.freeze_vision_encoder:
        return None, None

    # 使用无梯度上下文获取缓存的图像特征（节省计算和内存）
    with torch.no_grad():
        observation_features = policy.actor.encoder.get_cached_image_features(observations)
        next_observation_features = policy.actor.encoder.get_cached_image_features(next_observations)

    return observation_features, next_observation_features


def use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    """
    判断是否使用线程模式进行并发处理。

    Args:
        cfg (TrainRLServerPipelineConfig): 训练配置对象

    Returns:
        bool: 如果配置指定使用线程模式则返回True，否则返回False
    """
    return cfg.policy.concurrency.learner == "threads"


def check_nan_in_transition(
    observations: torch.Tensor,
    actions: torch.Tensor,
    next_state: torch.Tensor,
    raise_error: bool = False,
) -> bool:
    """
    检查转换数据中是否存在NaN值。

    NaN值可能导致训练不稳定或失败，因此在训练前检测它们很重要。

    Args:
        observations (torch.Tensor): 观察张量字典
        actions (torch.Tensor): 动作张量
        next_state (torch.Tensor): 下一步状态张量字典
        raise_error (bool): 如果为True，检测到NaN时抛出ValueError

    Returns:
        bool: 如果检测到NaN值返回True，否则返回False
    """
    nan_detected = False

    # 检查观察中的NaN值
    for key, tensor in observations.items():
        if torch.isnan(tensor).any():
            logging.error(f"observations[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in observations[{key}]")

    # 检查下一步状态中的NaN值
    for key, tensor in next_state.items():
        if torch.isnan(tensor).any():
            logging.error(f"next_state[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in next_state[{key}]")

    # 检查动作中的NaN值
    if torch.isnan(actions).any():
        logging.error("actions contains NaN values")
        nan_detected = True
        if raise_error:
            raise ValueError("NaN detected in actions")

    return nan_detected


def push_actor_policy_to_queue(parameters_queue: Queue, policy: nn.Module):
    """
    将Actor策略参数推送到队列，供Actor服务器获取和更新。

    该函数将策略网络的参数（Actor网络和可选的离散Critic网络）序列化
    后放入队列，以便Actor服务器可以定期获取最新的策略参数。

    Args:
        parameters_queue (Queue): 用于传递参数的队列
        policy (nn.Module): 包含Actor和Critic的策略对象
    """
    logging.debug("[LEARNER] Pushing actor policy to the queue")

    # 创建一个字典来保存所有的状态字典
    # 将Actor参数移到CPU以避免GPU内存问题
    state_dicts = {"policy": move_state_dict_to_device(policy.actor.state_dict(), device="cpu")}

    # 如果存在离散Critic网络，也将其加入状态字典
    if hasattr(policy, "discrete_critic") and policy.discrete_critic is not None:
        state_dicts["discrete_critic"] = move_state_dict_to_device(
            policy.discrete_critic.state_dict(), device="cpu"
        )
        logging.debug("[LEARNER] Including discrete critic in state dict push")

    # 将状态字典序列化为字节并放入队列
    state_bytes = state_to_bytes(state_dicts)
    parameters_queue.put(state_bytes)


def process_interaction_message(
    message, interaction_step_shift: int, wandb_logger: WandBLogger | None = None
):
    """
    处理单个交互消息，进行反序列化、步数偏移调整和日志记录。

    该函数将从Actor接收到的交互消息（字节数组）转换为Python对象，
    对交互步数进行调整以保持与检查点状态的一致性，并可选地记录到WandB。

    Args:
        message: 从Actor接收到的字节数组格式的交互消息
        interaction_step_shift (int): 交互步数的偏移量，用于恢复训练时的步数对齐
        wandb_logger (WandBLogger | None): WandB日志记录器，如果为None则不记录

    Returns:
        dict: 处理后的交互消息字典
    """
    # 将字节数组反序列化为Python对象
    message = bytes_to_python_object(message)
    # 调整交互步数以保持与检查点状态的一致性
    message["Interaction step"] += interaction_step_shift

    # 如果提供了WandB日志记录器，则记录该消息
    if wandb_logger:
        wandb_logger.log_dict(d=message, mode="train", custom_step_key="Interaction step")

    return message


def process_transitions(
    transition_queue: Queue,
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer,
    device: str,
    dataset_repo_id: str | None,
    shutdown_event: any,
):
    """
    处理队列中所有可用的转换数据。

    该函数从Actor接收转换数据，移动到指定设备，检查NaN值，
    然后添加到在线回放缓冲区。如果是干预数据，也添加到离线缓冲区。

    Args:
        transition_queue (Queue): 用于接收Actor转换数据的队列
        replay_buffer (ReplayBuffer): 添加转换数据的在线回放缓冲区
        offline_replay_buffer (ReplayBuffer): 添加转换数据的离线回放缓冲区
        device (str): 将转换数据移动到的设备
        dataset_repo_id (str | None): 数据集仓库ID
        shutdown_event (Event): 用于通知关闭的事件
    """
    # 处理队列中所有可用的转换数据
    while not transition_queue.empty() and not shutdown_event.is_set():
        transition_list = transition_queue.get()
        # 将字节数组反序列化为转换列表
        transition_list = bytes_to_transitions(buffer=transition_list)

        for transition in transition_list:
            # 将转换数据移动到指定设备（CPU/GPU）
            transition = move_transition_to_device(transition=transition, device=device)

            # 跳过包含NaN值的转换（可能导致训练不稳定）
            if check_nan_in_transition(
                observations=transition["state"],
                actions=transition[ACTION],
                next_state=transition["next_state"],
            ):
                logging.warning("[LEARNER] NaN detected in transition, skipping")
                continue

            # 将转换添加到在线回放缓冲区
            replay_buffer.add(**transition)

            # 如果是干预数据，也添加到离线缓冲区
            # 干预数据是人类操作员接管控制的数据，非常有价值
            if dataset_repo_id is not None and transition.get("complementary_info", {}).get(
                TeleopEvents.IS_INTERVENTION
            ):
                offline_replay_buffer.add(**transition)


def process_interaction_messages(
    interaction_message_queue: Queue,
    interaction_step_shift: int,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,
) -> dict | None:
    """
    处理队列中所有可用的交互消息。

    该函数从队列中获取所有交互消息，对每条消息进行处理（反序列化、
    步数偏移调整、日志记录），并返回最后处理的消息。

    Args:
        interaction_message_queue (Queue): 用于接收交互消息的队列
        interaction_step_shift (int): 交互步数的偏移量
        wandb_logger (WandBLogger | None): 用于跟踪进度的日志记录器
        shutdown_event (Event): 用于通知关闭的事件

    Returns:
        dict | None: 最后处理的交互消息，如果没有处理任何消息则返回None
    """
    last_message = None
    # 处理队列中所有可用的交互消息
    while not interaction_message_queue.empty() and not shutdown_event.is_set():
        message = interaction_message_queue.get()
        # 处理单条交互消息
        last_message = process_interaction_message(
            message=message,
            interaction_step_shift=interaction_step_shift,
            wandb_logger=wandb_logger,
        )

    return last_message


if __name__ == "__main__":
    train_cli()
    logging.info("[LEARNER] main finished")
