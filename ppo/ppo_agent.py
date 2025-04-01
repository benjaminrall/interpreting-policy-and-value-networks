from gymnasium.vector import VectorEnv
import torch
import gymnasium as gym
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import numpy as np
from torch.types import Tensor
from torch.optim import Adam
from .config import PPOConfig
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter

class PPOAgent(nn.Module):
    """Base PPO Agent class. Works for environments with discrete actions."""

    def __init__(self, envs: VectorEnv | None) -> None:
        """Creates a new PPO Agent for the given vector environment."""
        super().__init__()
        self.global_step = 0
        self.updates_completed = 0
        self.envs = envs
        self._construct_critic()
        self._construct_actor()

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

    def _construct_critic(self) -> None:
        """Constructs and stores the critic network for the agent."""
        input_size = np.array(self.envs.single_observation_space.shape).prod()
        self.critic = nn.Sequential(
            self._init_layer(nn.Linear(input_size, 64)),
            nn.Tanh(),
            self._init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            self._init_layer(nn.Linear(64, 1), weight_std=1)
        )

    def _construct_actor(self) -> None:
        """Constructs and stores the actor network for the agent."""
        input_size = np.array(self.envs.single_observation_space.shape).prod()
        output_size = self.envs.single_action_space.n
        self.actor = nn.Sequential(
            self._init_layer(nn.Linear(input_size, 64)),
            nn.Tanh(),
            self._init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            self._init_layer(nn.Linear(64, output_size), weight_std=0.01)
        )

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

    def anneal_lr(self, update: int, total_updates: int) -> None:
        """Anneals the optimiser's learning rate linearly for the given update."""
        frac = 1 - (update - 1) / total_updates
        self.optimiser.param_groups[0]['lr'] = frac * self.cfg.learning_rate

    def get_action_logits(self, observation: Tensor) -> Tensor:
        return self.actor(observation)

    def get_value(self, observation: Tensor) -> Tensor:
        return self.critic(observation)

    def get_action(self, observation: Tensor) -> Tensor:
        return Categorical(logits=self.actor(observation)).sample()

    def get_action_and_value(
        self, observation: Tensor, action: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Bundles actor and critic evaluation to return a sampled action and value.

        Parameters
        ----------
        observation : Tensor
            Current observation from the environment.
        action : Tensor | None, optional
            Specified action to be taken in the current state, by default None,
            in which case the action is sampled from the actor network for rollout

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            - Sampled action to be taken given the observation (or the specified value)
            - Log probability of the returned action
            - Entropies of the action probability distribution
            - Critic value of the observation
        """
        # Gets un-normalised action probabilites from the actor network
        logits = self.actor(observation)

        # Uses a categorical distribution over the actor's output for the actions
        probs = Categorical(logits=logits)

        # If no action was specified, sample it from the probabilities
        action = action if action is not None else probs.sample()

        # Returns the tuple of action details and value
        return action, probs.log_prob(action), probs.entropy(), self.critic(observation)

    def compute_gae(
        self,
        next_obs: Tensor,
        next_done: Tensor,
        storage: dict[str, Tensor],        
        device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Performs generalised advantage estimation."""
        with torch.no_grad():
            # Variables to store the advantages for each environment and the previous lambda value
            advantages = torch.zeros_like(storage['rewards']).to(device)
            last_lambda = 0

            # Initial values assuming continuation from the last step in the current batch
            next_non_terminal = 1 - next_done
            next_values = self.get_value(next_obs).reshape(1, -1)

            # Iterates backwards through the steps recorded for the current batch to compute advantages
            for t in reversed(range(self.cfg.n_steps)):
                # Updates the next values for the current step being evaluated
                if t < self.cfg.n_steps - 1:
                    next_non_terminal = 1 - storage['dones'][t + 1]
                    next_values = storage['values'][t + 1]

                # Calculates GAE using the equations from the paper by Schulman et al. (2015)
                delta = storage['rewards'][t] + self.cfg.gamma * next_values * next_non_terminal - storage['values'][t]
                advantages[t] = last_lambda = delta + self.cfg.gamma * self.cfg.gae_lambda * next_non_terminal * last_lambda

            # Computes returns for the batch
            returns = advantages + storage['values']

        return advantages, returns
    
    def init_data_store(self, device: torch.device) -> dict[str, Tensor]:
        """Initialises the data store used to store all required results from policy rollout."""
        batch_shape = (self.cfg.n_steps, self.cfg.n_environments)
        return {
            'observations': torch.zeros(batch_shape + self.envs.single_observation_space.shape).to(device).double(),
            'actions': torch.zeros(batch_shape + self.envs.single_action_space.shape).to(device).double(),
            'log_probs': torch.zeros(batch_shape).to(device).double(),
            'rewards': torch.zeros(batch_shape).to(device).double(),
            'dones': torch.zeros(batch_shape).to(device).double(),
            'values': torch.zeros(batch_shape).to(device).double(),
        }
    
    def get_batch_data(
        self,
        advantages: Tensor, 
        returns: Tensor, 
        storage: dict[str, Tensor], 
    ) -> dict[str, Tensor]:
        """Gets the complete flattened data for a batch from its rollout storage."""
        return {
            'observations': storage['observations'].reshape((-1, *self.envs.single_observation_space.shape)),
            'log_probs': storage['log_probs'].reshape(-1),
            'actions': storage['actions'].reshape((-1, *self.envs.single_action_space.shape)),
            'advantages': advantages.reshape(-1),
            'returns': returns.reshape(-1),
            'values': storage['values'].reshape(-1)
        }


    def save(self):
        """
        Saves the current model state
        """
        checkpoint = {
            'global_step': self.global_step,
            'updates_completed': self.updates_completed,
            'agent_state_dict': self.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict(),
        }

        path = f'checkpoints/{self.cfg.run_name}-{self.updates_completed}'
        torch.save(checkpoint, path)


    @classmethod
    def load(cls, cfg: PPOConfig, path: str, envs: VectorEnv) -> 'PPOAgent':
        agent = cls(envs)
        device = agent.get_device()

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        agent.load_state_dict(checkpoint['agent_state_dict'])

        agent.optimiser = Adam(agent.parameters(), lr=cfg.learning_rate, eps=cfg.adam_eps)
        agent.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

        agent.global_step = checkpoint['global_step']
        agent.updates_completed = checkpoint['updates_completed']

        return agent.to(device)


    def policy_rollout(
        self,
        next_obs: Tensor,
        next_done: Tensor,
        storage: dict[str, Tensor],
        global_step: int,
        device: torch.device,
        writer: SummaryWriter,
    ) -> tuple[Tensor, Tensor, int]:
        """
        Performs the policy rollout phase, interacting with the 
        environment to collect data needed for training.

        Parameters
        ----------
        next_obs : Tensor
            Initial observation for each environment.
        next_done : Tensor
            Initial termination state for each environment.
        storage : dict[str, Tensor]
            Dictionary to store rollout outputs.
        global_step : int
            Current global step number for metrics tracking.
        device : torch.device
            Device to use for torch. 
        writer : SummaryWriter
            Tensorboard writer for tracking metrics.

        Returns
        -------
        tuple[Tensor, Tensor, int]
            Tuple containing the following three outputs:
            - Initial observation for each environment for the next training step.
            - Initial termination state for each environment for the next training step.
            - Updated global step value.
        """
        # Iterates each environment for the number of steps specified in the config
        for step in range(self.cfg.n_steps):
            # Increment global steps per environment and store the current observation
            global_step += self.cfg.n_environments
            storage['observations'][step] = next_obs
            storage['dones'][step] = next_done

            # Use the agent to get actions, probabilities, and value estimates for the current state
            with torch.no_grad():
                action, log_prob, _, value = self.get_action_and_value(next_obs)
                storage['values'][step] = value.flatten()
            storage['actions'][step] = action
            storage['log_probs'][step] = log_prob

            # Step the environments using the sampled actions, and store their results
            next_obs, reward, terminate, truncate, info = self.envs.step(action.cpu().numpy())
            storage['rewards'][step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.tensor(next_obs).to(device).double()
            next_done = (torch.tensor(terminate + truncate) > 0).type(torch.int32).to(device)

            # If an environment finished, log its stats to Tensorboard
            if next_done.any():
                for i in range(self.cfg.n_environments):
                    if next_done[i] and 'episode' in info:
                        writer.add_scalar(
                            'charts/episodic_return',
                            info['episode']['r'][i],
                            global_step
                        )
                        writer.add_scalar(
                            'charts/episodic_length',
                            info['episode']['l'][i],
                            global_step,
                        )
                
        return next_obs, next_done, global_step
    
    def optimise(
        self,
        batch_data: dict[str, Tensor],
        global_step: int,
        device: torch.device,
        writer: SummaryWriter
    ) -> None:
        """Optimises the agent's networks based on a given set of complete batch data."""
        # Loops for each update epoch
        for epoch in range(self.cfg.update_epochs):
            # Randomly shuffles batch indices
            batch_idx = torch.randperm(self.cfg.batch_size).to(device)

            # Processes and updates the agent for each mini-batch
            for start in range(0, self.cfg.batch_size, self.cfg.minibatch_size):
                # Gets data indices for each mini-batch
                end = start + self.cfg.minibatch_size
                minibatch_idx = batch_idx[start:end]

                # Extracts observed minibatch data
                observations = batch_data['observations'].index_select(0, minibatch_idx)
                actions = batch_data['actions'].index_select(0, minibatch_idx)
                log_probs = batch_data['log_probs'].index_select(0, minibatch_idx)
                advantages = batch_data['advantages'].index_select(0, minibatch_idx)
                returns = batch_data['returns'].index_select(0, minibatch_idx)
                values = batch_data['values'].index_select(0, minibatch_idx)

                # Gets probabilites and values for the updated policy and value networks
                _, new_log_probs, entropy, new_values = self.get_action_and_value(observations, actions)

                # Calculates the ratio between the probabilities given for the action by the new and old policies
                ratio = (new_log_probs - log_probs).exp()

                # Performs advantage normalisation
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Calculates policy loss using clipped policy objective from PPO paper
                unclipped_policy_loss = advantages * ratio
                clipped_policy_loss = advantages * torch.clamp(ratio, 1 - self.cfg.clip_coefficient, 1 + self.cfg.clip_coefficient)
                policy_loss = torch.min(unclipped_policy_loss, clipped_policy_loss).mean()

                # Calculates value loss using MSE
                value_loss = ((new_values.view(-1) - returns) ** 2.0).mean()

                # Calculates entropy loss, which should be maximised to encourage exploration
                entropy_loss = entropy.mean()

                # Combines policy, value, and entropy loss into the final optimisation objective
                loss = -policy_loss - self.cfg.entropy_coefficient * entropy_loss + self.cfg.value_coefficient * value_loss

                # Performs backpropagation to find the gradients of this loss w.r.t the agent's parameters
                self.optimiser.zero_grad()
                loss.backward()

                # Clips gradients to avoid them exploding
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.max_grad_norm)

                # Applies the gradients using the optimiser
                self.optimiser.step()

        # Adds tracked values to the Tensorboard writer
        writer.add_scalar('charts/learning_rate', self.optimiser.param_groups[0]['lr'], global_step)
        writer.add_scalar('losses/policy_loss', policy_loss.item(), global_step)
        writer.add_scalar('losses/value_loss', value_loss.item(), global_step)
        writer.add_scalar('losses/entropy_loss', entropy_loss.item(), global_step)
        writer.add_scalar('losses/combined_loss', loss.item(), global_step)

    def train(self, cfg: PPOConfig, writer: SummaryWriter):
        # Gets the device to use for torch
        device = self.get_device()

        # Sets the config file and optimiser for training
        self.cfg = cfg
        self.to(device).double()
        self.optimiser = Adam(self.parameters(), lr=cfg.learning_rate, eps=cfg.adam_eps)

        # Initialises the data store used to track all relevant values during training
        storage = self.init_data_store(device)

        # Tensors to keep track of the current state and termination criteria during training 
        next_obs = torch.tensor(self.envs.reset(seed=cfg.seed)[0]).to(device).double()
        next_done = torch.zeros(cfg.n_environments).to(device)

        # Extra values for tracking run performance and performing learning rate annealing etc.
        total_updates = (cfg.total_timesteps // cfg.batch_size)

        # Main training loop
        for update in tqdm(range(1 + self.updates_completed, total_updates + 1), initial=self.updates_completed, total=total_updates):
            # Anneals the learning rate of the optimiser linearly
            if cfg.anneal_lr:
                self.anneal_lr(update, total_updates)

            # Performs policy rollout on the environments
            next_obs, next_done, self.global_step = self.policy_rollout(
                next_obs,
                next_done,
                storage,
                self.global_step,
                device,
                writer
            )

            # Performs generalised advantage estimation and flattens this rollout's data into a batch
            advantages, returns = self.compute_gae(next_obs, next_done, storage, device)
            batch_data = self.get_batch_data(advantages, returns, storage)

            # Optimises the agent based on this collected data
            self.optimise(batch_data, self.global_step, device, writer)

            # Saves checkpoints of the agent
            self.updates_completed += 1
            if cfg.save_checkpoints and self.updates_completed % cfg.checkpoint_updates == 0:
                self.save()
        
        self.save()