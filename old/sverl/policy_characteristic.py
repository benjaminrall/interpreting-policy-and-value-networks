from gymnasium.vector import VectorEnv
import torch
import gymnasium as gym
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import numpy as np
from torch.types import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset, random_split

class PolicyCharacteristic(nn.Module):
    """Model to approximate the policy characteristic function for SVERL."""

    def __init__(self, output_size=4) -> None:
        super().__init__()
        self.output_size = output_size
        self._construct_network()
        self.epochs_completed = 0
        self.optimiser = None
        self.scheduler = None

    @staticmethod
    def get_device() -> torch.device:
        """
        Gets the device to be used by torch.
        - `cuda` for NVIDIA and AMD
        - `mps` for Apple
        - `cpu` otherwise
        """
        return torch.device(
            'cuda' if torch.cuda.is_available() else 
            'mps' if torch.mps.is_available() else 
            'cpu'
        )

    @staticmethod
    def _init_layer(
        layer: nn.Linear, weight_std=np.sqrt(2), bias=0
    ) -> nn.Linear:
        """
        Initialises a linear neural network layer.
        Uses orthogonal initialisation for weights, and constant initialisation for biases.
        """
        torch.nn.init.orthogonal_(layer.weight, weight_std)
        torch.nn.init.constant_(layer.bias, bias)
        return layer
    
    def _construct_network(self):
        self.network = nn.Sequential(
            self._init_layer(nn.Conv2d(4, 64, 8, stride=4)),
            nn.ReLU(),
            self._init_layer(nn.Conv2d(64, 128, 4, stride=2)),
            nn.ReLU(),\
            self._init_layer(nn.Conv2d(128, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            self._init_layer(nn.Linear(64 * 7 * 7, 1024)),
            nn.ReLU(),
            self._init_layer(nn.Linear(1024, 512)),
            nn.ReLU(),
            self._init_layer(nn.Linear(512, self.output_size), weight_std=0.01),
        )
        # self.network = nn.Sequential(
        #     self._init_layer(nn.Conv2d(4, 32, 8, stride=4)),
        #     nn.ReLU(),
        #     self._init_layer(nn.Conv2d(32, 64, 4, stride=2)),
        #     nn.ReLU(),
        #     self._init_layer(nn.Conv2d(64, 64, 3, stride=1)),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     self._init_layer(nn.Linear(64 * 7 * 7, 512)),
        #     nn.ReLU(),
        #     self._init_layer(nn.Linear(512, self.output_size), weight_std=0.01),
        # )

    def get_action_logits(self, observation: Tensor) -> Tensor:
        return self.network(observation)
    
    def anneal_lr(self, update: int, total_updates: int) -> None:
        """Anneals the optimiser's learning rate linearly for the given update."""
        frac = 1 - (update - 1) / total_updates
        self.optimiser.param_groups[0]['lr'] = frac * 0.001

    def save(self):
        """
        Saves the current model state
        """
        checkpoint = {
            'epochs_completed': self.epochs_completed,
            'agent_state_dict': self.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }

        path = f'checkpoints/policy-characteristic-{self.epochs_completed}'
        torch.save(checkpoint, path)


    @classmethod
    def load(cls, path: str) -> 'PolicyCharacteristic':
        agent = cls()
        device = agent.get_device()

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        agent.load_state_dict(checkpoint['agent_state_dict'])

        agent.optimiser = Adam(agent.parameters(), lr=0.001)
        agent.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

        agent.scheduler = ReduceLROnPlateau(agent.optimiser, mode='min', factor=0.8, patience=3)
        agent.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        agent.epochs_completed = checkpoint['updates_completed']

        return agent.to(device)

    def policy_rollout(self, device, obs, agent, envs: VectorEnv, samples):
        sampled_obs = np.zeros((samples,) + obs.shape[1:])
        sampled_actions = np.zeros((samples, agent.output_size))
        for i in tqdm(range(samples // envs.num_envs)):
            with torch.no_grad():
                action, _, _, value = agent.get_action_and_value(obs)
                action_logits = agent.get_action_logits(obs)
            sampled_obs[i*envs.num_envs:(i+1)*envs.num_envs] = obs.cpu().numpy()
            sampled_actions[i*envs.num_envs:(i+1)*envs.num_envs] = action_logits.cpu().numpy()
        
            next_obs, _, _, _, _ = envs.step(action.cpu().numpy())
            obs = torch.tensor(next_obs).to(device).double()
        
        sampled_obs = torch.Tensor(sampled_obs).to(device).double()
        sampled_actions = torch.Tensor(sampled_actions).to(device).double()
        return sampled_obs, sampled_actions

    def train(self, agent, envs, writer: SummaryWriter, train_size=27000, val_size=3000):
        # Initialise optimiser and LR scheduler
        device = self.get_device()
        self.to(device).double()
        if not self.optimiser:
            self.optimiser = Adam(self.parameters(), lr=0.003)
        if not self.scheduler:
            self.scheduler = ReduceLROnPlateau(self.optimiser, mode='min', factor=0.8, patience=5)

        # Generate validation data
        val_xs = torch.Tensor(np.load('data/observations.npy'))
        val_ys = torch.Tensor(np.load('data/actions.npy'))
        data = TensorDataset(val_xs, val_ys)
        other_data, val_data = random_split(data, [len(data) - val_size, val_size])
        del val_xs, val_ys, data, other_data

        val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
        val_masks = [torch.rand(obs.shape) < 0.5 for obs, _ in val_loader]
        
        EPOCHS = 500
        obs = torch.tensor(envs.reset()[0]).to(device).double()
        for epoch in range(1 + self.epochs_completed, EPOCHS + 1):
            print(f"EPOCH {epoch}")
            print("Sampling training data")
            train_x, train_y = self.policy_rollout(device, obs, agent, envs, train_size)
            train_data = TensorDataset(train_x, train_y)
            train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

            print("Training...")
            # Loop through mini batches
            for observations, action_probs in tqdm(train_loader):
                # For each mini batch:
                # - Get mask
                mask = torch.rand(observations.shape) < 0.5
                
                # - Get masked observation
                observations[mask] = 0

                # - Get masked output
                predictions = self.get_action_logits(observations)

                # - Calculate loss
                loss = torch.square(action_probs - predictions).mean()
                
                # - Step optimiser
                self.optimiser.zero_grad()
                loss.backward()

                self.optimiser.step()

                writer.add_scalar('losses/training_loss', loss.item())

            del train_x, train_y, train_data, train_loader
            torch.cuda.empty_cache()

            # Measure validation loss
            total_loss = 0
            for i, (observations, action_probs) in enumerate(val_loader):
                observations[val_masks[i]] = 0
                observations = observations.to(device).double()
                action_probs = action_probs.to(device).double()
                action_probs = torch.softmax(action_probs, dim=-1)
                predictions = torch.softmax(self.get_action_logits(observations), dim=-1)
                total_loss += torch.square(action_probs - predictions).sum().item()
                del observations, action_probs
            total_loss /= len(val_data)
            writer.add_scalar('losses/validation_loss', total_loss, epoch)

            # Step the LR scheduler
            self.scheduler.step(total_loss)
            writer.add_scalar('charts/learning_rate', self.optimiser.param_groups[0]['lr'], epoch)

            self.epochs_completed += 1
            if self.epochs_completed % 50 == 0:
                self.save()