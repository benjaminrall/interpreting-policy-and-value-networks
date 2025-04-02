import random
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import time
import argparse
import numpy as np
import gymnasium as gym
from gymnasium.vector import VectorEnv
from ppo.ppo_agent import PPOAgent
from ppo.continuous_ppo_agent import ContinuousPPOAgent
from ppo.atari_ppo_agent import AtariPPOAgent
from env_utils import init_envs, init_atari_envs
import wandb
from sverl.policy_characteristic import PolicyCharacteristic
import os
from ppo.config import PPOConfig

from torch.utils.data import DataLoader, TensorDataset

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
    parser.add_argument(
        '--checkpoint', '-ch',
        type=str,
        default=None,
        nargs='?',
        help='Checkpoint file.',
    )
    args = parser.parse_args()

    # Loads the config from the specified yaml file
    cfg = PPOConfig.from_yaml(args.config_file)

    # Sets random seeds to ensure run is deterministic
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    envs = init_atari_envs(cfg, False)
    agent = AtariPPOAgent.load(cfg, args.checkpoint, envs)
    device = agent.get_device()
    agent.to(device).double()

    # Sets up Tensorboard writer for the run
    run_name = f'{'PolicyCharacteristic'}_{int(time.time())}'
    wandb.init(
        project="Year 3 Project",
        entity=None,
        sync_tensorboard=True,
        name=run_name,
        monitor_gym=True,
        save_code=True
    )

    writer = SummaryWriter(f'runs/{run_name}')
    # writer.add_text('hyperparameters', cfg.to_table())


    # Initialises the environments and trains an agent
    model = PolicyCharacteristic()
    # observations = torch.Tensor(np.load('observations.npy')).to(model.get_device()).double()

    # actions = torch.Tensor(np.load('actions.npy')).to(model.get_device()).double()

    # data = TensorDataset(observations, actions)
    model.train(agent, envs, writer)

    # Closes the Tensorboard writer
    writer.close()
