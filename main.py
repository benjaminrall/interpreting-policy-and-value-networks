from src import Trainer
import argparse
import wandb

if __name__ == '__main__':

    wandb.login(key='fde56b5f00cdbd73fda76d359e6eb51a5d9f3fcf')

    # Reads command line argument for the config file path or checkpoint path
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--config', '-c',
        type=argparse.FileType('r'),
        help='Config file in YAML format.',
    )
    group.add_argument(
        '--checkpoint', '-ch',
        type=str,
        help='Checkpoint file for a training run.'
    )
    args = parser.parse_args()

    # Instantiates a trainer from the config or checkpoint
    if args.config:
        trainer = Trainer.from_yaml(args.config)
    else:
        trainer = Trainer.load_checkpoint(args.checkpoint)

    # Trains the model
    trainer.train()
