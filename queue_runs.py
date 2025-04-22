import subprocess
import argparse
from pathlib import Path

def run_configs(config_paths):
    for config_path in config_paths:
        print(f"\n=== Running config: {config_path} ===")
        subprocess.run(["py", "main.py", "-c", str(config_path)])

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run multiple config files.")
    parser.add_argument('configs', metavar='config', type=Path, nargs='+',
                        help="A list of config file paths to run")
    
    args = parser.parse_args()
    
    # Run the config files
    run_configs(args.configs)

if __name__ == "__main__":
    main()