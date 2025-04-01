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
from env_utils import init_envs, init_atari_playback_env, init_atari_envs
import wandb
import os
import pygame
import cv2
from tqdm import tqdm

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

    # Initialises the environments and trains an agent
    envs = init_atari_envs(cfg, False)
    agent = AtariPPOAgent.load(cfg, args.checkpoint, envs)
    

    # Gets the device and casts the agent to it
    device = agent.get_device()
    agent.to(device).double()

    # Gets the first observation
    obs = torch.tensor(envs.reset()[0]).to(device).double()

    SAMPLES = 30000
    sample_observations = np.zeros((SAMPLES,) + obs.shape[1:])
    sample_actions = np.zeros((SAMPLES, agent.output_size))
    sample_values = np.zeros((SAMPLES, 1))
    for i in tqdm(range(SAMPLES // cfg.n_environments)):
        with torch.no_grad():
            action, _, _, value = agent.get_action_and_value(obs)
            action_logits = agent.get_action_logits(obs)

        sample_observations[i*cfg.n_environments:(i+1)*cfg.n_environments] = obs.cpu().numpy()
        sample_actions[i*cfg.n_environments:(i+1)*cfg.n_environments] = action_logits.cpu().numpy()
        sample_values[i*cfg.n_environments:(i+1)*cfg.n_environments] = value.cpu().numpy()
        
        next_obs, reward, terminate, truncate, info = envs.step(action.cpu().numpy())
        obs = torch.tensor(next_obs).to(device).double()

    np.save('observations', sample_observations)
    np.save('actions', sample_actions)
    np.save('values', sample_values)

    envs.close()
    