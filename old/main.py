import random
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import time
import argparse
import numpy as np
import gymnasium as gym
from gymnasium.vector import VectorEnv
from ppo.config import PPOConfig
from ppo.ppo_agent import PPOAgent
from ppo.continuous_ppo_agent import ContinuousPPOAgent
from ppo.atari_ppo_agent import AtariPPOAgent
from env_utils import init_envs, init_atari_envs
import wandb
import os

def get_agent(envs: VectorEnv) -> PPOAgent:
    """Returns the correct PPO agent for the given environments based on the their type."""
    return ContinuousPPOAgent(envs) if isinstance(envs.single_action_space, gym.spaces.Box) else PPOAgent(envs)

if __name__ == '__main__':
    # Reads command line argument for config file path
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-file', '-c',
        type=argparse.FileType('r'),
        default=None,
        nargs='?',
        help='Config file in YAML format.',
    )
    args = parser.parse_args()

    # Loads the config from the specified yaml file
    cfg = PPOConfig.from_yaml(args.config_file)

    # Sets random seeds to ensure run is deterministic
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    
    # Sets up Tensorboard writer for the run
    run_name = f'{cfg.run_name}_{int(time.time())}'
    if cfg.track_wandb:
        wandb.init(
            project=cfg.wandb_project_name,
            entity=cfg.wandb_entity,
            sync_tensorboard=True,
            config=vars(cfg),
            name=run_name,
            monitor_gym=True,
            save_code=True
        )

    writer = SummaryWriter(f'runs/{run_name}')
    writer.add_text('hyperparameters', cfg.to_table())

    # Initialises the environments and trains an agent
    envs = init_atari_envs(cfg) if cfg.atari else init_envs(cfg)
    agent = AtariPPOAgent(envs) if cfg.atari else get_agent(envs)
    agent.train(cfg, writer)
    envs.close()

    # Closes the Tensorboard writer
    writer.close()
