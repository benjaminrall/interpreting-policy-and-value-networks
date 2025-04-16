import random
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import time
import argparse
import numpy as np
import gymnasium as gym
from gymnasium.vector import VectorEnv
import wandb
import os
import pygame
import cv2
from src import Trainable
from src.agents import Agent
from src.utils import get_device
from src.env_builders import EnvBuilder
from src.configs import EnvConfig

# Constants
WIN_WIDTH = 640
WIN_HEIGHT = 840
FRAMERATE = 50

# Pygame Setup
win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Agent Testing")
clock = pygame.time.Clock()

if __name__ == '__main__':
    # Reads command line argument for config file path
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint', '-ch',
        type=str,
        default=None,
        nargs='?',
        help='Checkpoint file for the agent to be tested.',
    )
    args = parser.parse_args()

    # Initialises the environments and trains an agent
    device = get_device()
    agent: Agent = Trainable.load_checkpoint(args.checkpoint).to(device)

    # Adjusts EnvConfig for testing purposes
    env_config: EnvConfig = agent.cfg.environment
    env_config.num_envs = 1
    env_config.record = False

    # Builds the environment to test the agent on
    envs = EnvBuilder.build(env_config)
    obs = torch.tensor(envs.reset()[0]).to(device)
    
    surface = pygame.Surface((WIN_WIDTH, WIN_HEIGHT))
    running = True
    playing = False
    stepping = False
    render = None
    while running:
        dt = clock.tick(FRAMERATE) * 0.001

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                envs.close()
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    playing = not playing
                elif event.key == pygame.K_r:
                    envs = EnvBuilder.build(env_config)
                    obs = torch.tensor(envs.reset()[0]).to(device)
                elif event.key == pygame.K_RIGHT:
                    stepping = True
                elif event.key == pygame.K_s:
                    test_save = {
                        'render': render,
                        'state': obs.cpu().numpy(),
                    }
                    torch.save(test_save, f"data/test/breakout/{int(time.time())}")
                    
        if playing or stepping:
            action = agent.get_action(obs)
            next_obs, reward, terminate, truncate, info = envs.step(action.cpu().numpy())
            obs = torch.tensor(next_obs).to(device)
            done = terminate and truncate
        stepping = False
        render = np.swapaxes(envs.unwrapped.render()[0], 0, 1)
        pygame.surfarray.blit_array(surface, cv2.resize(render, (WIN_HEIGHT, WIN_WIDTH), cv2.INTER_NEAREST))
        win.blit(surface, (0, 0))
        pygame.display.update()

    envs.close()
