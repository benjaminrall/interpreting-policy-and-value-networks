from src.sverl import SVERLFunction
from src.utils import get_device
import torch
from torch.optim import Adam
from torch.types import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from src.sverl.masking import Masker
from src.sverl import Characteristic
from src import Trainable
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.configs.sverl import ShapleyConfig
    from src import Trainer

class Shapley(SVERLFunction):
    """SVERL shapley value approximator."""

    def __init__(self, cfg: 'ShapleyConfig', state: dict = None):
        super().__init__(cfg, state)
        self.cfg = cfg
        self.device = get_device()
        self.to(self.device)

        # Loads the characteristic function to be used from the given checkpoint path
        self.characteristic: Characteristic = Trainable.load_checkpoint(cfg.characteristic_checkpoint)

        # Initialises the Adam optimiser and learning rate schedular
        self.optimiser = Adam(self.parameters(), lr=cfg.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimiser, mode='min', factor=cfg.anneal_factor, patience=cfg.anneal_patience)

        # Initialises training tracking params
        self.epochs_completed = 0

        # Loads state if it was provided
        if state is not None:
            self.optimiser.load_state_dict(state['optimiser'])
            self.scheduler.load_state_dict(state['scheduler'])
            self.epochs_completed = state['epochs_completed']

    def get_state_dict(self):
        state_dict = super().get_state_dict()
        state_dict['optimiser'] = self.optimiser.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        state_dict['epochs_completed'] = self.epochs_completed
        return state_dict
    
    def infer(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            # Gets the target function output
            target_output = self.target(x)
            if self.cfg.target == 'actor':
                target_output = torch.softmax(target_output, dim=-1)

            # Gets the characteristic function output
            masked = self.characteristic.masker.masked_like(x)
            cf_output = self.characteristic.infer(masked)

            # Calculates the contributions of each pixel from the shapley model
            contributions = self.model(x)
            dim = tuple(range(1, masked.dim()))
            contribution_sum = torch.sum(contributions, dim=dim)

            # Calculates the normalisation factor to apply to all contributions for the state
            norm_factor = (1 / np.prod(x.shape[1:])) * (target_output - cf_output - contribution_sum)
            norm_dim = [x.shape[0]] + [1] * (x.dim() - 1) + [target_output.shape[-1]]
        return contributions + norm_factor.view(norm_dim)
    
    def train(self, trainer: 'Trainer'):
        # Generates the validation data for tracking the training run
        val_xs, val_masks = self.generate_validation_data()

        for epoch in tqdm(range(1 + self.epochs_completed, self.cfg.epochs + 1), initial=self.epochs_completed, total=self.cfg.epochs):
            # Gets state samples for the current epoch
            samples = self.state_sampler.sample(self.cfg.samples_per_epoch, self.cfg.minibatch_size)

            for i, x in enumerate(samples):
                # Generates the random mask for this batch
                mask = torch.rand(x.shape) < 0.5

                # Applies the mask to the state observation
                self.characteristic.masker.mask(x, mask)
                masked = self.characteristic.masker.masked_like(x)

                # Calculates the relevant characteristic function results 
                part_1 = self.characteristic.infer(x)
                part_2 = self.characteristic.infer(masked)

                # Gets the relevant outputs from the shapley model
                results = self.model(x)
                masked_results = mask.unsqueeze(-1).to(self.device) * results
                dim = tuple(range(1, masked_results.dim() - 1))
                part_3 = masked_results.sum(dim=dim)

                # EXPERIMENT 3
                # action = i % 4
                # part_1_a = part_1[..., action]
                # part_2_a = part_2[..., action]
                # part_3_a = part_3[..., action]

                # Calculates MSE loss
                loss = torch.square(part_1 - part_2 - part_3).mean()

                # Steps the optimiser
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                # Logs loss to the trainer
                trainer.log('losses/training_loss', loss.item())

            # Measures validation loss
            total_loss = 0
            for i, x in enumerate(val_xs):
                # Applies the mask to the state observation
                mask = val_masks[i]
                self.characteristic.masker.mask(x, mask)
                masked = self.characteristic.masker.masked_like(x)

                # Calculates the relevant characteristic function results 
                part_1 = self.characteristic.infer(x)
                part_2 = self.characteristic.infer(masked)

                # Gets the relevant outputs from the shapley model
                with torch.no_grad():
                    results = self.model(x)
                    masked_results = mask.unsqueeze(-1).to(self.device) * results
                    dim = tuple(range(1, masked_results.dim() - 1))
                    part_3 = masked_results.sum(dim=dim)

                    # Calculates MSE loss
                    total_loss += torch.square(part_1 - part_2 - part_3).sum().item()

            # Calculates and logs the average validation loss
            total_loss /= self.cfg.validation_samples
            trainer.log('losses/validation_loss', total_loss, epoch)

            # Steps the LR scheduler
            self.scheduler.step(total_loss)
            trainer.log('charts/learning_rate', self.optimiser.param_groups[0]['lr'], epoch)
            
            # Reports epochs completed to the trainer for checkpoint saving
            self.epochs_completed += 1
            trainer.update(self.epochs_completed)

        # Saves a final checkpoint for the end of training
        trainer.save_checkpoint(self.epochs_completed)
