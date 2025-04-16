import subprocess
from pathlib import Path

CONFIG_DIR = Path("configs/breakout/sverl")

for config_path in ['configs/breakout/sverl/policy-characteristic-1LL.yaml', 'configs/breakout/sverl/policy-shapley-1L.yaml']:#sorted(CONFIG_DIR.glob("*.yaml")):
    print(f"\n=== Running config: {config_path} ===")
    subprocess.run(["py", "main.py", '-c' + str(config_path)])