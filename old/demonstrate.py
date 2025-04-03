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

# Constants
WIN_WIDTH = 640
WIN_HEIGHT = 840
FRAMERATE = 50

# Pygame Setup
win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Agent Demonstration")
clock = pygame.time.Clock()

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
    agent = AtariPPOAgent.load(cfg, args.checkpoint, init_atari_envs(cfg))
    

    # Gets the device and casts the agent to it
    device = agent.get_device()
    agent.to(device).double()

    # Gets the first observation
    env = init_atari_playback_env(cfg)
    obs = torch.tensor(env.reset()[0]).to(device).double()
    
    surface = pygame.Surface((WIN_WIDTH, WIN_HEIGHT))
    running = True
    playing = False
    stepping = False
    while running:
        dt = clock.tick(FRAMERATE) * 0.001

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                env.close()
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    playing = not playing
                elif event.key == pygame.K_r:
                    env = init_atari_playback_env(cfg)
                    obs = torch.tensor(env.reset()[0]).to(device).double()
                elif event.key == pygame.K_RIGHT:
                    stepping = True
                elif event.key == pygame.K_s:
                    np.save(f"states/breakout/{int(time.time())}", obs.unsqueeze(0).cpu().numpy())
                    
        if playing or stepping:
            action = agent.get_action(obs.unsqueeze(0))
            next_obs, reward, terminate, truncate, info = env.step(action[0].cpu().numpy())
            obs = torch.tensor(next_obs).to(device).double()
            done = terminate and truncate
        stepping = False
        
        pygame.surfarray.blit_array(surface, cv2.resize(np.swapaxes(env.unwrapped.render(), 0, 1), (WIN_HEIGHT, WIN_WIDTH), cv2.INTER_NEAREST))
        win.blit(surface, (0, 0))
        pygame.display.update()

    env.close()
