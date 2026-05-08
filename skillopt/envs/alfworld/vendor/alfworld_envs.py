# Vendored from SkillRL (Apache-2.0 License)
# Original: agent_system/environments/env_package/alfworld/envs.py
# Modified: imports use pip-installed alfworld package instead of vendored copy.

import os
import multiprocessing as mp
import traceback
import yaml
import gymnasium as gym
import numpy as np

from alfworld.agents.environment import get_environment


def load_config_file(path):
    assert os.path.exists(path), f"Invalid config file: {path}"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config


def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward


class AlfworldWorker:
    """Stateful worker that holds one ALFWorld sub-environment."""

    def __init__(self, config, seed, base_env, gamefile=None):
        if gamefile:
            base_env.game_files = [gamefile]
            if hasattr(base_env, "num_games"):
                base_env.num_games = 1
        self.env = base_env.init_env(batch_size=1)
        self.env.seed(seed)

    def step(self, action):
        actions = [action]
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        return obs, scores, dones, infos

    def reset(self):
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        return obs, infos


def _worker_loop(cmd_q, result_q, config, seed, is_train, eval_dataset, gamefile):
    """Run one ALFWorld environment in a child process."""
    try:
        env_type = config['env']['type']
        base_env = get_environment(env_type)(
            config,
            train_eval='train' if is_train else eval_dataset,
        )
        worker = AlfworldWorker(config, seed, base_env, gamefile)
        result_q.put((True, "ready"))
    except BaseException:
        result_q.put((False, traceback.format_exc()))
        return

    while True:
        cmd, payload = cmd_q.get()
        if cmd == "close":
            result_q.put((True, None))
            return
        try:
            if cmd == "reset":
                result = worker.reset()
            elif cmd == "step":
                result = worker.step(payload)
            else:
                raise ValueError(f"Unknown ALFWorld worker command: {cmd}")
            result_q.put((True, result))
        except BaseException:
            result_q.put((False, traceback.format_exc()))


class _ProcessWorker:
    """Small stdlib actor wrapper for one environment process."""

    def __init__(self, ctx, config, seed, is_train, eval_dataset, gamefile=None):
        self.cmd_q = ctx.Queue(maxsize=1)
        self.result_q = ctx.Queue(maxsize=1)
        self.process = ctx.Process(
            target=_worker_loop,
            args=(self.cmd_q, self.result_q, config, seed, is_train, eval_dataset, gamefile),
        )
        self.process.start()
        ok, payload = self.result_q.get()
        if not ok:
            self.close(kill=True)
            raise RuntimeError(f"Failed to start ALFWorld worker:\n{payload}")

    def send(self, cmd, payload=None):
        self.cmd_q.put((cmd, payload))

    def recv(self):
        ok, payload = self.result_q.get()
        if not ok:
            raise RuntimeError(f"ALFWorld worker failed:\n{payload}")
        return payload

    def close(self, kill=False):
        if self.process.is_alive() and not kill:
            try:
                self.send("close")
                self.recv()
            except Exception:
                kill = True
        if kill and self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1)
        self.cmd_q.close()
        self.result_q.close()


class AlfworldEnvs(gym.Env):
    """Vectorized ALFWorld environment using local process workers."""

    def __init__(self, alf_config_path, seed, env_num, group_n,
                 resources_per_worker, is_train=True, env_kwargs=None, gamefiles=None):
        super().__init__()
        if env_kwargs is None:
            env_kwargs = {}

        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n
        self.gamefiles = list(gamefiles or [])
        if self.gamefiles and len(self.gamefiles) != self.num_processes:
            raise ValueError(
                f"Expected {self.num_processes} gamefiles, got {len(self.gamefiles)}"
            )

        start_method = os.environ.get("ALFWORLD_WORKER_START_METHOD") or None
        ctx = mp.get_context(start_method) if start_method else mp.get_context()
        self.workers = []
        for i in range(self.num_processes):
            worker_gamefile = self.gamefiles[i] if self.gamefiles else None
            worker = _ProcessWorker(
                ctx,
                config,
                seed + (i // self.group_n),
                is_train,
                eval_dataset,
                worker_gamefile,
            )
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes

        for i, worker in enumerate(self.workers):
            worker.send("step", actions[i])
        results = [worker.recv() for worker in self.workers]

        text_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]
            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)
            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        image_obs_list = None
        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        for worker in self.workers:
            worker.send("reset")
        results = [worker.recv() for worker in self.workers]

        text_obs_list = []
        info_list = []

        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        image_obs_list = None
        return text_obs_list, image_obs_list, info_list

    @property
    def get_admissible_commands(self):
        return self.prev_admissible_commands

    def close(self):
        for worker in self.workers:
            worker.close()


def build_alfworld_envs(alf_config_path, seed, env_num, group_n,
                        resources_per_worker, is_train=True, env_kwargs=None, gamefiles=None):
    """Build vectorized ALFWorld environments."""
    return AlfworldEnvs(
        alf_config_path, seed, env_num, group_n,
        resources_per_worker, is_train, env_kwargs, gamefiles,
    )
