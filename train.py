from src import Trainer

with open('configs/breakout/ppo.yaml', 'r') as f:
    trainer = Trainer.from_yaml(f)
    
trainer.train()