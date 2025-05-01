from src import Trainer
# with open('configs/mastermind/agent.yaml') as f:
#     t = Trainer.from_yaml(f)


# t.train()
# print(t.objective.Q_table)

# import torch
# qs = torch.load('qs.pt', weights_only=False)
# print(qs)
# t.objective.Q_table = qs

# t.save_checkpoint(10000)
t = Trainer.load_checkpoint('models/testing/mastermind.pt')
print(t.objective.Q_table)