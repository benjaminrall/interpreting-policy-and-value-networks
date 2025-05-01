from src import Trainer

with open('configs/grid/agent.yaml') as f:
    t = Trainer.from_yaml(f)

t.train()
print(t.objective.Q_table)