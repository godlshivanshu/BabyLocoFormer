# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class RslRlPpoActorCriticTransformerXLCfg(RslRlPpoActorCriticCfg):
    """Specialized config with Transformer-XL specific hyper-parameters."""

    class_name: str = "ActorCriticTransformerXL"
    transformer_model_dim: int = 64
    transformer_num_layers: int = 4
    transformer_num_heads: int = 4
    transformer_ff_multiplier: float = 4.0
    transformer_dropout: float = 0.0
    memory_length: int = 32

# @configclass
# class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     num_steps_per_env = 24
#     max_iterations = 5000
#     save_interval = 500
#     experiment_name = ""  # same as task name
#     empirical_normalization = False
#     policy = RslRlPpoActorCriticCfg(
#         init_noise_std=1.0,
#         actor_hidden_dims=[512, 256, 128],
#         critic_hidden_dims=[512, 256, 128],
#         activation="elu",
#     )
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=1.0,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.01,
#         num_learning_epochs=5,
#         num_mini_batches=4,
#         learning_rate=1.0e-3,
#         schedule="fixed",
#         gamma=0.99,
#         lam=0.95,
#         desired_kl=0.01,
#         max_grad_norm=1.0,
#     )

@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 10000
    save_interval = 300
    experiment_name = ""
    policy = RslRlPpoActorCriticTransformerXLCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[128, 64],
        activation="elu",

        transformer_model_dim=64,
        transformer_num_layers = 3,
        transformer_num_heads=4,
        transformer_ff_multiplier=4.0,
        transformer_dropout=0.0,
        memory_length=32,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1.0e-3,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
