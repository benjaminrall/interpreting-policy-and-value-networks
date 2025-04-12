from src import Trainer
import argparse

if __name__ == '__main__':
    # Reads command line argument for the config file path or checkpoint path
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--config-file', '-c',
        type=argparse.FileType('r'),
        help='Config file in YAML format.',
    )
    group.add_argument(
        '--checkpoint-file', '-ch',
        type=str,
        help='Checkpoint file for a training run.'
    )
    args = parser.parse_args()

    # Instantiates a trainer from the config or checkpoint
    if args.config_file:
        trainer = Trainer.from_yaml(args.config_file)
    else:
        trainer = Trainer.load_checkpoint(args.checkpoint_file)

    # Trains the model
    trainer.train()
