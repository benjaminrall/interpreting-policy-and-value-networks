from src.sverl import SVERLFunction
from src.utils import get_device
import torch
from torch.optim import Adam
from torch.types import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from src.sverl.masking import Masker

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.configs.sverl import CharacteristicConfig
    from src import Trainer

class Characteristic(SVERLFunction):
    """SVERL characteristic function approximator."""

    def __init__(self, cfg: 'CharacteristicConfig', state: dict = None):
        super().__init__(cfg, state)
        self.cfg = cfg
        self.device = get_device()
        self.to(self.device)

        # Gets the masker instance for masking states
        if cfg.mask_kwargs is None:
            cfg.mask_kwargs = {}
        self.masker = Masker.from_name(cfg.masker, agent=self.agent, **cfg.mask_kwargs)

        # Initialises the Adam optimiser and learning rate schedular
        self.optimiser = Adam(self.parameters(), lr=cfg.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimiser, mode='min', factor=0.8, patience=5)

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
            predictions = self.model(x)
            if self.cfg.target == 'actor':
                predictions = torch.softmax(predictions, dim=-1)
        return predictions

    def train(self, trainer: 'Trainer'):
        val_samples, val_masks = self.generate_validation_data()

        for epoch in tqdm(range(1 + self.epochs_completed, self.cfg.epochs + 1), initial=self.epochs_completed, total=self.cfg.epochs):
            # Gets state samples for the current epoch
            samples = self.state_sampler.sample(self.cfg.samples_per_epoch, self.cfg.batch_size)

            for i, (x, y) in enumerate(samples):
                # Generates the random mask for this batch
                mask = torch.rand(x.shape) < 0.5

                # Applies the mask to the state observation
                self.masker.mask(x, mask)

                # Gets masked output from the characteristic model
                predictions = self.model(x)

                # Calculates MSE loss
                loss = torch.square(y - predictions).mean()

                # Steps the optimiser
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                # Logs loss to the trainer
                trainer.log('losses/training_loss', loss.item())

            # Measures validation loss
            total_loss = 0
            for i, (x, y) in enumerate(val_samples):
                self.masker.mask(x, val_masks[i])
                with torch.no_grad():
                    predictions = self.model(x)
                    if self.cfg.target == 'actor':
                        y = torch.softmax(y, dim=-1)
                        predictions = torch.softmax(predictions, dim=-1)
                    total_loss += torch.square(y - predictions).sum().item()

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
