# Interpreting Policy and Value Networks: An Explainability-Based Comparison in Deep Reinforcement Learning Agents

This repository contains the code used to generate results for my third year dissertation.

To run a model, use the `main.py` script, specifying either a config file (`-c`) or checkpoint (`-ch`).
For example:

```bash
python main.py -c configs/breakout/agents/ppo.yaml
```

will train a PPO agent to play Breakout, using the configuration parameters specified in the provided YAML file.
