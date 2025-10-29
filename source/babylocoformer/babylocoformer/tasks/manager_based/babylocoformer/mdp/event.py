from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

import omni.usd
import torch
from pxr import UsdGeom

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


class compute_nominal_heights(ManagerTermBase):
    """Cache nominal base heights by reading link geometry directly from the USD stage."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._stage = omni.usd.get_context().get_stage()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | Sequence[int] | None,
        asset_cfg: SceneEntityCfg,
        segments: Sequence[Mapping[str, object]] | None = None,
        base_offset: float | int | None = 0.0,
    ):
        asset: Articulation = env.scene[asset_cfg.name]
        indices = self._normalize_env_ids(env, env_ids)
        buffer = self._ensure_buffer(env, asset)

        if not segments:
            default_root_state = asset.data.default_root_state
            if default_root_state is None:
                raise RuntimeError(
                    "Unable to compute nominal heights because the articulation does not expose a default root state."
                )
            buffer[indices] = default_root_state[indices, 2]
            return

        stage = self._stage
        link_paths = asset.root_physx_view.link_paths
        body_to_index = {name: idx for idx, name in enumerate(asset.data.body_names)}
        offset_value = float(base_offset) if base_offset is not None else 0.0

        for env_id in indices.tolist():
            nominal_height = offset_value
            for spec in segments:
                body_name = str(spec["body"])
                rel_path = str(spec.get("geometry_path", ""))
                multiplier = float(spec.get("multiplier", 1.0))

                body_idx = body_to_index[body_name]
                link_path = self._resolve_link_path(link_paths, env_id, body_idx)
                prim_path = self._resolve_prim_path(stage, link_path, rel_path)
                nominal_height += multiplier * self._read_cylinder_height(stage, prim_path)

            buffer[env_id] = nominal_height

    @staticmethod
    def _normalize_env_ids(
        env: ManagerBasedEnv, env_ids: torch.Tensor | Sequence[int] | None
    ) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=env.device, dtype=torch.long)
        if isinstance(env_ids, slice):
            return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
        return torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)

    @staticmethod
    def _resolve_link_path(link_paths: Sequence, env_id: int, body_idx: int) -> str:
        try:
            env_paths = link_paths[env_id]  # type: ignore[index]
        except (IndexError, TypeError):
            env_paths = link_paths[0]  # type: ignore[index]
        return str(env_paths[body_idx])

    def _resolve_prim_path(self, stage, link_path: str, relative_path: str) -> str:
        if not relative_path:
            return link_path.rstrip("/")

        candidate = f"{link_path.rstrip('/')}/{relative_path.lstrip('/')}"
        prim = stage.GetPrimAtPath(candidate)
        if prim and prim.IsValid():
            return candidate

        mesh_candidate = f"{link_path.rstrip('/')}/mesh_0/{relative_path.lstrip('/')}"
        prim = stage.GetPrimAtPath(mesh_candidate)
        if prim and prim.IsValid():
            return mesh_candidate

        raise RuntimeError(f"Prim '{candidate}' not found while computing nominal height.")

    def _read_cylinder_height(self, stage, prim_path: str) -> float:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Prim '{prim_path}' not found while computing nominal height.")
        if not prim.IsA(UsdGeom.Cylinder):
            raise RuntimeError(f"Prim '{prim_path}' is not a UsdGeom.Cylinder.")
        height = UsdGeom.Cylinder(prim).GetHeightAttr().Get()
        if height is None:
            raise RuntimeError(f"Cylinder prim '{prim_path}' does not define a height attribute.")
        return float(height)

    @staticmethod
    def _ensure_buffer(env: ManagerBasedEnv, asset: Articulation) -> torch.Tensor:
        dtype = (
            asset.data.default_root_state.dtype
            if asset.data.default_root_state is not None
            else torch.float32
        )
        if not hasattr(env, "_nominal_heights") or env._nominal_heights.shape[0] != env.num_envs:
            env._nominal_heights = torch.zeros(env.num_envs, dtype=dtype, device=env.device)
        else:
            env._nominal_heights = env._nominal_heights.to(device=env.device, dtype=dtype)
        return env._nominal_heights
