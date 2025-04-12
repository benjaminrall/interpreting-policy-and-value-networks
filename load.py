
from dataclasses import dataclass
from src.configs import TrainerConfig
from src import Trainer
from src.agents import PPO
from src.utils import to_nested_dict
with open('configs/breakout/breakout_new.yaml', 'r') as f:
    cfg = TrainerConfig.from_yaml(f)

checkpoint = 'checkpoints\\breakout-ppo\\0.pt'
trainer = Trainer.load_checkpoint(checkpoint)
print(trainer.objective)