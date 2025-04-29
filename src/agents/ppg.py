import torch
from torch.optim import Adam
from torch.types import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from src.agents import PPO
from src.utils import get_device
from tqdm import tqdm
from torch.distributions import Categorical

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.configs.agent import PPGConfig
    from src import Trainer

class PPG(PPO):
    """PPG agent."""

    def __init__(self, cfg: 'PPGConfig', state: dict = None):
        super().__init__(cfg, state)
        self.cfg = cfg
        
    def init_data_store(self) -> dict[str, Tensor]:
        """Initialises the data store used to store all required results from policy rollout."""
        storage = super().init_data_store()
        aux_batch_shape = (self.cfg.n_steps, self.cfg.aux_batch_rollouts)
        storage.update({
            'aux_observations': torch.zeros(aux_batch_shape + self.envs.single_observation_space.shape),
            'aux_returns': torch.zeros(aux_batch_shape)
        })
        return storage
    
    def get_batch_data(
        self,
        advantages: Tensor, 
        returns: Tensor, 
        storage: dict[str, Tensor], 
    ) -> dict[str, Tensor]:
        """Gets the complete flattened data for a batch from its rollout storage."""
        batch_data = super().get_batch_data(advantages, returns, storage)
        if self.cfg.adv_norm_fullbatch:
            mean =  batch_data['advantages'].mean()
            std =  batch_data['advantages'].std()
            batch_data['advantages'] = (batch_data['advantages'] - mean) / (std + 1e-8)
        return batch_data
    
    def optimise(self, batch_data: dict[str, Tensor], trainer: 'Trainer') -> None:
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

    @staticmethod
    def to_batches(tensor: Tensor):
        """Converts (minibatch, action, ...) tensors to (batch, ...) tensors."""
        return tensor.reshape((-1, *tensor.shape[2:]))

    def from_batches(self, tensor: Tensor):
        """Converts (batch, ...) tensors to (minibatch, action, ...) tensors."""
        return tensor.reshape((-1, self.cfg.n_auxiliary_rollouts, *tensor.shape[1:]))

    def get_policy_buffer(self, storage: dict[str, Tensor]) -> Tensor:
        # Computes and stores the current policy for all states in the replay buffer
        aux_idx = torch.arange(self.cfg.aux_batch_rollouts)
        aux_policy = torch.zeros((self.cfg.n_steps, self.cfg.aux_batch_rollouts, self.envs.single_action_space.n))
        for i, start in enumerate(range(0, self.cfg.aux_batch_rollouts, self.cfg.n_auxiliary_rollouts)):
            end = start + self.cfg.n_auxiliary_rollouts
            # print(aux_idx)

            aux_minibatch_idx = aux_idx[start:end]
            # print(aux_minibatch_idx)

            aux_observations = storage['aux_observations'].index_select(1, aux_minibatch_idx).to(self.device)
            aux_observations_shape = aux_observations.shape
            aux_observations = self.to_batches(aux_observations)

            # print(aux_observations_shape)
            # print(aux_observations.shape)

            with torch.no_grad():
                action_logits = self.get_action_logits(aux_observations).cpu().clone()

            # print(action_logits.shape)
            # print(aux_policy.shape)
            # print(aux_minibatch_idx)
            # print(self.from_batches(action_logits, aux_observations_shape[:2]).shape)
            # print(aux_policy[aux_minibatch_idx, :].shape)
            aux_policy[:, aux_minibatch_idx] = self.from_batches(action_logits)
            
            del aux_observations
        return aux_policy

    def train(self, trainer: 'Trainer') -> None:
        # Initialises the data store used to track all relevant values during training
        storage = self.init_data_store()

        # Tensors to keep track of the current state and termination criteria during training 
        next_obs = torch.tensor(self.envs.reset(seed=trainer.cfg.seed)[0]).to(self.device)
        next_done = torch.zeros(self.cfg.environment.num_envs).to(self.device)

        for phase in tqdm(range(1 + self.updates_completed, self.cfg.n_phases + 1), initial=self.updates_completed, total=self.cfg.n_phases):
            # POLICY PHASE
            for policy_update in range(1, self.cfg.n_policy_iterations + 1):
                if self.cfg.anneal_lr:
                    self.anneal_lr((phase - 1) * self.cfg.n_policy_iterations + policy_update, self.cfg.n_iterations)

                # Performs policy rollout on the environments
                next_obs, next_done = self.policy_rollout(
                    next_obs,
                    next_done,
                    storage,
                    trainer,
                )

                # Performs generalised advantage estimation and flatten's this rollout's data into a batch
                advantages, returns = self.compute_gae(next_obs, next_done, storage)
                batch_data = self.get_batch_data(advantages, returns, storage)

                # Optimises the agent based on this collected data
                self.optimise(batch_data, trainer)

                # Stores rollouts for sampling during auxiliary stage
                n_envs = self.cfg.environment.num_envs
                storage_slice = slice(n_envs * (policy_update - 1), n_envs * policy_update)
                storage['aux_observations'][:, storage_slice] = storage['observations'].cpu().clone()
                storage['aux_returns'][:, storage_slice] = returns.cpu().clone()
            
            # AUXILIARY PHASE
            aux_policy: Tensor = self.get_policy_buffer(storage)
            self.optimiser.zero_grad()


            for aux_update in range(self.cfg.auxiliary_update_epochs):
                batch_idx = torch.randperm(self.cfg.aux_batch_rollouts, dtype=int)

                for i, start in enumerate(range(0, self.cfg.aux_batch_rollouts, self.cfg.n_auxiliary_rollouts)):
                    end = start + self.cfg.n_auxiliary_rollouts
                    aux_minibatch_idx = batch_idx[start:end]
                    # print(aux_minibatch_idx)
                    aux_observations = self.to_batches(storage['aux_observations'].index_select(1, aux_minibatch_idx).to(self.device))
                    aux_returns = self.to_batches(storage['aux_returns'].index_select(1, aux_minibatch_idx).to(self.device))
                    # print(storage['aux_observations'].shape)
                    # print(aux_policy.shape)
                    new_action_logits = self.actor.forward(aux_observations)
                    new_policy = Categorical(logits=new_action_logits)

                    a = aux_policy[:, aux_minibatch_idx].to(self.device)

                    old_action_logits = self.to_batches(a)
                    old_policy = Categorical(logits=old_action_logits)

                    new_values = self.critic.forward(aux_observations).view(-1)
                    new_aux_values: Tensor = self.actor.forward_value(aux_observations).view(-1)

                    kl_loss = torch.distributions.kl_divergence(old_policy, new_policy).mean()

                    real_value_loss = 0.5 * ((new_values.view(-1) - aux_returns) ** 2.0).mean()
                    aux_value_loss = 0.5 * ((new_aux_values.view(-1) - aux_returns) ** 2.0).mean()
                    joint_loss = aux_value_loss + self.cfg.beta_clone * kl_loss

                    loss = (joint_loss + real_value_loss) / self.cfg.n_aux_grad_accum
                    loss.backward()

                    if (i + 1) % self.cfg.n_aux_grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.max_grad_norm)
                        self.optimiser.step()
                        self.optimiser.zero_grad()
                    
                    del aux_observations, aux_returns
        
                trainer.log("losses/aux/kl_loss", kl_loss.mean().item(), self.global_step)
                trainer.log("losses/aux/aux_value_loss", aux_value_loss.item(), self.global_step)
                trainer.log("losses/aux/real_value_loss", real_value_loss.item(), self.global_step)

            # Saves checkpoints of the agent
            self.updates_completed += 1
            trainer.update(self.updates_completed)

        # Saves a final checkpoint for the end of training
        trainer.save_checkpoint(self.updates_completed)