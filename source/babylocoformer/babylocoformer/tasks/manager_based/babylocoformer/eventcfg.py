# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    leg_length_init = EventTerm(
        func=mdp.compute_nominal_heights,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "segments": [
                {"body": "RL_thigh", "geometry_path": "collisions/mesh_0/cylinder", "multiplier": 0.75},
                {"body": "RL_calf", "geometry_path": "collisions/mesh_0/cylinder", "multiplier": 0.95},
            ],
        },
    )
    
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )

    # Mass randomization for different parts
    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (3.0, 7.0),
            "operation": "abs",
            "recompute_inertia": True,
        },
    )
    randomize_hip_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_hip"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "abs",
            "recompute_inertia": True,
        },
    )
    randomize_thigh_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_thigh"),
            "mass_distribution_params": (1.0, 3.0),
            "operation": "abs",
            "recompute_inertia": True,
        },
    )
    randomize_calf_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_calf"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "abs",
            "recompute_inertia": True,
        },
    )
    randomize_foot_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "mass_distribution_params": (0.1, 0.3),
            "operation": "abs",
            "recompute_inertia": True,
        },
    )

    # CoM jitter for all links
    jitter_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "com_range": {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "z": (-0.005, 0.005)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

    # periodic reset of the robot state without terminating the episode
    # this is useful for continuous data collection where the robot is reset to a new state
    # without ending the MDP.
    # periodic_cart_reset = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="interval",
    #     interval_range_s=(2.0, 2.0),  # reset every 2 seconds
    #     is_global_time=False,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
    #         "position_range": (-1.0, 1.0),
    #         "velocity_range": (-0.5, 0.5),
    #     },
    # )

    # periodic_pole_reset = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="interval",
    #     interval_range_s=(2.0, 2.0),   # reset every 2 seconds
    #     is_global_time=False,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
    #         "position_range": (-0.25 * math.pi, 0.25 * math.pi),
    #         "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
    #     },
    # )
