import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os
import random


class DoubleInvertedPendulumEnv(gym.Env):
    def __init__(self, render=False, max_steps=500):
        super().__init__()
        self.render = render
        self.max_steps = max_steps
        self.current_step = 0
        if self.render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, -9.81, 0)

        # Load model
        self.model_path = os.path.join(
            os.path.dirname(__file__),
            "..",  # go up one directory from dip_env/
            "models",
            "double_pendulum.urdf",
        )
        self.model_path = os.path.abspath(self.model_path)

        self.pendulum_id = p.loadURDF(self.model_path, [0, 0, 0.2])

        # Action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-5.0]), high=np.array([5.0]), dtype=np.float32
        )
        obs_high = np.array([np.pi, np.pi, 10.0, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

    def seed(self, seed=None):
        # Set the seed for reproducibility. Be permissive about input types
        # and ensure the seed is an integer in [0, 2**32 - 1].
        if seed is None:
            normalized = None
        else:
            # If seed is array-like (e.g. numpy scalar/array or list/tuple),
            # try to extract a single integer value.
            try:
                # numpy types have item() method; plain ints will pass through
                if hasattr(seed, "item"):
                    seed_val = seed.item()
                else:
                    seed_val = seed
                seed_int = int(seed_val)
            except Exception:
                raise ValueError(f"Invalid seed value: {seed!r}")

            # Coerce into valid uint32 range
            normalized = seed_int % (2**32)

        self.seed_value = normalized
        self.np_random, used_seed = gym.utils.seeding.np_random(normalized)
        # gym.utils.seeding.np_random may return a very large int (especially
        # when normalized is None). Numpy's legacy RandomState requires a
        # seed in [0, 2**32 - 1], so coerce it into that range before using.
        if used_seed is None:
            return [None]

        used_seed_int = int(used_seed)
        used_seed32 = used_seed_int % (2**32)
        np.random.seed(used_seed32)
        random.seed(used_seed32)
        # Keep a stable python int for self.seed_value so super().reset gets
        # a native int in the acceptable range.
        self.seed_value = used_seed32
        return [used_seed32]

    def reset(self, *, seed=None, options=None):
        # Normalize and set the seed first so that gymnasium receives a
        # native Python int or None (gym validates the seed type/value).
        self.seed(seed)
        super().reset(seed=self.seed_value)
        self.current_step = 0
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.pendulum_id = p.loadURDF(self.model_path, [0, 0, 0.2])
        p.setJointMotorControlArray(
            self.pendulum_id, [0, 1], p.VELOCITY_CONTROL, forces=[0, 0]
        )

        # Small random perturbation
        theta1, theta2 = np.random.uniform(-0.1, 0.1, size=2)
        p.resetJointState(self.pendulum_id, 0, theta1)
        p.resetJointState(self.pendulum_id, 1, theta2)

        obs = self._get_obs()
        info = {"reset_info": "Environment reset with random initial state"}
        return obs, info

    def step(self, action):
        self.current_step += 1

        # Clip action to range
        torque = float(np.clip(action, self.action_space.low, self.action_space.high))
        p.setJointMotorControl2(self.pendulum_id, 0, p.TORQUE_CONTROL, force=torque)
        p.setJointMotorControl2(self.pendulum_id, 1, p.TORQUE_CONTROL, force=0.0)

        p.stepSimulation()

        obs = self._get_obs()
        theta1, theta2, dtheta1, dtheta2 = obs

        reward = self._compute_reward(obs)

        terminated = bool(abs(theta1) > np.pi or abs(theta2) > np.pi)

        truncated = self.current_step >= self.max_steps

        is_success = bool(abs(theta1) < 0.1 and abs(theta2) < 0.1)

        info = {
            "theta1": theta1,
            "theta2": theta2,
            "dtheta1": dtheta1,
            "dtheta2": dtheta2,
            "reward": reward,
            "success": is_success,
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        s1 = p.getJointState(self.pendulum_id, 0)
        s2 = p.getJointState(self.pendulum_id, 1)
        return np.array([s1[0], s2[0], s1[1], s2[1]], dtype=np.float32)

    def _compute_reward(self, obs):
        # Upright: theta1 ≈ 0, theta2 ≈ 0, and low angular velocity
        theta1, theta2, dtheta1, dtheta2 = obs
        return -(theta1**2 + theta2**2 + 0.1 * dtheta1**2 + 0.1 * dtheta2**2)

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect()
