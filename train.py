import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from dip_env.dip_env import DoubleInvertedPendulumEnv
import logging

# -------------------------------
# Configure logging
# -------------------------------
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------------------
# Create the environment
# -------------------------------

SEED = 42
# Device to use for training: default to CPU. Override with DEVICE env var (e.g. DEVICE=cuda)
DEVICE = os.environ.get("DEVICE", "cpu")


def create_env(env_class=DoubleInvertedPendulumEnv, render=False):
    return Monitor(env_class(render=render))

train_env = create_env(render=True)
eval_env = create_env(render=False)


# Custom callback to stop training after a certain number of timesteps
class StopTrainingCallback(BaseCallback):
    def __init__(self, stop_timesteps, verbose=0):
        super(StopTrainingCallback, self).__init__(verbose)
        self.stop_timesteps = stop_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.stop_timesteps:
            if self.verbose > 0:
                logger.info(
                    f"Reached {self.num_timesteps} timesteps, stopping training."
                )
            return False  # Returning False stops training
        return True


# Custom TensorBoard logging callback
class TensorboardCallback(BaseCallback):
    def __init__(self, eval_freq=500, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.writer = None  # Will be initialized in _on_training_start

    def _on_training_start(self) -> None:
        log_dir = os.path.join(self.logger.dir or "./logs", "custom")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard logs will be saved to {log_dir}")

    def _on_step(self) -> bool:
        if self.writer is None:
            logger.error("TensorBoard writer is not initialized.")
            return False

        if self.n_calls % self.eval_freq == 0:
            self.logger.record("custom/num_timesteps", self.num_timesteps)

        info = self.locals.get("infos", [{}])[0]

        for key in ["theta1", "theta2", "dtheta1", "dtheta2", "reward", "success"]:
            if key in info:
                self.writer.add_scalar(
                    f"custom/{key}", float(info[key]), self.num_timesteps
                )

        return True

    def _on_training_end(self) -> None:
        if self.writer is not None:
            self.writer.close()
            logger.info("TensorBoard writer closed.")


# -------------------------------
# Callbacks and logging
# -------------------------------
def create_callback(eval_env, total_timesteps):
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/results",
        eval_freq=500,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    stop_callback = StopTrainingCallback(stop_timesteps=total_timesteps, verbose=1)
    tensorboard_callback = TensorboardCallback(eval_freq=500, verbose=1)
    return [eval_callback, stop_callback, tensorboard_callback]


# -------------------------------
# Training Function
# -------------------------------
def train_model(env, total_timesteps, tb_log_name):
    callbacks = create_callback(eval_env, total_timesteps)
    model = PPO(
        "MlpPolicy",
        env,
        seed=SEED,
        device=DEVICE,
        verbose=1,
        tensorboard_log="./logs/tensorboard/",
    )
    logger.info("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=tb_log_name,
        progress_bar=True,
    )
    logger.info("Training completed.")
    return model


# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    total_timesteps = 100000
    model = train_model(train_env, total_timesteps, "PPO_DIP_1")

    # Load the best model and continue training
    best_model_path = os.path.join("./logs/best_model", "best_model.zip")
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path} for further training.")
        model = PPO.load(
            best_model_path,
            env=train_env,
            verbose=1,
            tensorboard_log="./logs/tensorboard/",
            device=DEVICE,
        )
        model = train_model(train_env, total_timesteps, "PPO_DIP_2")
    else:
        logger.warning(
            f"No best model found at {best_model_path}. Skipping loading step."
        )
