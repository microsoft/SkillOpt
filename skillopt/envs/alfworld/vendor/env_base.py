# Vendored from SkillRL (Apache-2.0 License)
# Original: agent_system/environments/base.py
# Trimmed to only include what ALFWorld needs.

from typing import List, Tuple, Dict, Any
import numpy as np
from collections import defaultdict


def to_numpy(data):
    """Convert data to numpy array."""
    # Lazy-check for torch.Tensor to avoid hard dependency on torch
    _torch_tensor = None
    try:
        import torch
        _torch_tensor = torch.Tensor
    except ImportError:
        pass

    if _torch_tensor is not None and isinstance(data, _torch_tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        pass
    elif isinstance(data, (int, float, bool, Tuple, List)):
        data = np.array(data)
    else:
        raise ValueError(f"Unsupported type: {type(data)})")
    return data


class EnvironmentManagerBase:
    """Base class for vectorized environment managers.

    Manages a set of parallel environments, handles action projection,
    observation post-processing, and history tracking.
    """

    def __init__(self, envs, projection_f, config):
        self.envs = envs
        self.projection_f = projection_f
        self.config = config

    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        return {'text': None, 'image': obs, 'anchor': None}, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_observations = {
            'text': None,
            'image': next_obs,
            'anchor': None,
        }
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)
        return next_observations, rewards, dones, infos

    def close(self) -> None:
        self.envs.close()

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        batch_size = len(total_batch_list)

        success = defaultdict(list)
        for bs in range(batch_size):
            self._process_batch(bs, total_batch_list, total_infos, success)
        assert len(success['success_rate']) == batch_size
        return {key: np.array(value) for key, value in success.items()}

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                return
