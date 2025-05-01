from src import Trainer, Trainable
from src.agents import QLearning, PPO
from src.sverl import Shapley
import torch
import numpy as np
import pickle

shapleys: Shapley = Trainable.load_checkpoint('checkpoints/mastermind-ps-3/200.pt')

print(shapleys.agent.Q_table)

e_obs = torch.Tensor([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                  [0, 0, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1],
                  [0, 0, 0, 1, 2, 1, 0, 0, -1, -1, -1, -1]]).to(shapleys.device)

outs = {}

for x_0 in e_obs:
    x = x_0.unsqueeze(0)
    with torch.no_grad():
        # Gets the target function output
        target_output = shapleys.target(x)
        if shapleys.cfg.target == 'actor':
            target_output = torch.softmax(target_output, dim=-1)
        print(target_output)
        # Gets the characteristic function output
        masked = shapleys.characteristic.masker.masked_like(x)
        cf_output = shapleys.characteristic.infer(masked)

        # Calculates the contributions of each pixel from the shapley model
        contributions = shapleys.model(x)
        dim = tuple(range(1, masked.dim() - 1))
        contribution_sum = torch.sum(contributions, dim=dim)

        # Calculates the normalisation factor to apply to all contributions for the state
        norm_factor = (1 / np.prod(x.shape[1:])) * (target_output - cf_output - contribution_sum)
    values = contributions + norm_factor
    outs[*(x_0.int().cpu().numpy())] = values.squeeze(0).cpu().numpy() 

with open('PolicyShapley2.pkl', 'wb') as f:
    pickle.dump(outs, f)
