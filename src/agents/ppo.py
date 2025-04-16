import torch
from torch.optim import Adam
from torch.types import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from src.agents import Agent
from src.utils import get_device
from tqdm import tqdm

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.configs.agent import PPOConfig
    from src import Trainer

class PPO(Agent):
    """PPO agent."""
    
    def __init__(self, cfg: 'PPOConfig', state: dict = None):
        super().__init__(cfg, state)
        self.cfg = cfg
        self.device = get_device()
        self.to(self.device)

        # Initialises the Adam optimiser for PPO
        self.optimiser = Adam(self.parameters(), lr=cfg.learning_rate, eps=cfg.adam_eps)
        
        # Initialises training tracking params
        self.global_step = 0
        self.updates_completed = 0

        # Loads state if it was provided
        if state is not None:
            self.optimiser.load_state_dict(state['optimiser'])
            self.global_step = state['global_step']
            self.updates_completed = state['updates_completed']

    def get_state_dict(self):
        state_dict = super().get_state_dict()
        state_dict['optimiser'] = self.optimiser.state_dict()
        state_dict['global_step'] = self.global_step
        state_dict['updates_completed'] = self.updates_completed
        return state_dict

    def anneal_lr(self, update: int, total_updates: int) -> None:
        """Anneals the optimiser's learning rate linearly for the given update."""
        frac = 1 - (update - 1) / total_updates
        self.optimiser.param_groups[0]['lr'] = frac * self.cfg.learning_rate

    def compute_gae(
        self,
        next_obs: Tensor,
        next_done: Tensor,
        storage: dict[str, Tensor],        
    ) -> tuple[Tensor, Tensor]:
        """Performs generalised advantage estimation."""
        with torch.no_grad():
            # Variables to store the advantages for each environment and the previous lambda value
            advantages = torch.zeros_like(storage['rewards']).to(self.device)
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

    
    def init_data_store(self) -> dict[str, Tensor]:
        """Initialises the data store used to store all required results from policy rollout."""
        batch_shape = (self.cfg.n_steps, self.cfg.environment.num_envs)
        return {
            'observations': torch.zeros(batch_shape + self.envs.single_observation_space.shape).to(self.device),
            'actions': torch.zeros(batch_shape + self.envs.single_action_space.shape).to(self.device),
            'log_probs': torch.zeros(batch_shape).to(self.device),
            'rewards': torch.zeros(batch_shape).to(self.device),
            'dones': torch.zeros(batch_shape).to(self.device),
            'values': torch.zeros(batch_shape).to(self.device),
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
        

    def policy_rollout(
        self,
        next_obs: Tensor,
        next_done: Tensor,
        storage: dict[str, Tensor],
        trainer: 'Trainer',
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
        trainer : Trainer
            Trainer for tracking metrics.

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
            self.global_step += self.cfg.environment.num_envs
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
            storage['rewards'][step] = torch.tensor(reward).to(self.device).view(-1)
            next_obs = torch.tensor(next_obs).to(self.device)
            next_done = (torch.tensor(terminate + truncate) > 0).type(torch.int32).to(self.device)

            # If an environment finished, log its stats to Tensorboard
            if next_done.any():
                for i in range(self.cfg.environment.num_envs):
                    if next_done[i] and 'episode' in info:
                        trainer.log(
                            'charts/episodic_return',
                            info['episode']['r'][i],
                            self.global_step
                        )
                        trainer.log(
                            'charts/episodic_length',
                            info['episode']['l'][i],
                            self.global_step,
                        )
                
        return next_obs, next_done
    

    def optimise(
        self,
        batch_data: dict[str, Tensor],
        trainer: 'Trainer'
    ) -> None:
        """Optimises the agent's networks based on a given set of complete batch data."""
        # Loops for each update epoch
        for epoch in range(self.cfg.update_epochs):
            # Randomly shuffles batch indices
            batch_idx = torch.randperm(self.cfg.batch_size).to(self.device)

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
                value_loss = 0.5 * ((new_values.view(-1) - returns) ** 2.0).mean()

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
        trainer.log('charts/learning_rate', self.optimiser.param_groups[0]['lr'], self.global_step)
        trainer.log('losses/policy_loss', policy_loss.item(), self.global_step)
        trainer.log('losses/value_loss', value_loss.item(), self.global_step)
        trainer.log('losses/entropy_loss', entropy_loss.item(), self.global_step)
        trainer.log('losses/combined_loss', loss.item(), self.global_step)


    def train(self, trainer: 'Trainer') -> None:
        # Initialises the data store used to track all relevant values during training
        storage = self.init_data_store()

        # Tensors to keep track of the current state and termination criteria during training 
        next_obs = torch.tensor(self.envs.reset(seed=trainer.cfg.seed)[0]).to(self.device)
        next_done = torch.zeros(self.cfg.environment.num_envs).to(self.device)

        # Extra values for tracking run performance and performing learning rate annealing etc.
        total_updates = self.cfg.total_timesteps // self.cfg.batch_size

        # Main training loop
        for update in tqdm(range(1 + self.updates_completed, total_updates + 1), initial=self.updates_completed, total=total_updates):
            # Anneals the learning rate of the optimiser linearly
            if self.cfg.anneal_lr:
                self.anneal_lr(update, total_updates)

            # Performs policy rollout on the environments
            next_obs, next_done = self.policy_rollout(
                next_obs,
                next_done,
                storage,
                trainer,
            )

            # Performs generalised advantage estimation and flattens this rollout's data into a batch
            advantages, returns = self.compute_gae(next_obs, next_done, storage)
            batch_data = self.get_batch_data(advantages, returns, storage)

            # Optimises the agent based on this collected data
            self.optimise(batch_data, trainer)

            # Saves checkpoints of the agent
            self.updates_completed += 1
            trainer.update(self.updates_completed)
        
        # Saves a final checkpoint for the end of training
        trainer.save_checkpoint(self.updates_completed)