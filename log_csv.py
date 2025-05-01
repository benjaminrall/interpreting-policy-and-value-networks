import pandas as pd
import wandb
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Upload episodic return data to W&B.")
    parser.add_argument("csv_path", help="Path to the CSV file.")
    parser.add_argument("--project", help="W&B project name.", default='Year 3 Project')
    parser.add_argument("--entity", help="W&B entity (user or team).", default=None)
    args = parser.parse_args()

    # Read the CSV
    try:
        df = pd.read_csv(args.csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Extract global_step
    if 'global_step' not in df.columns:
        print("CSV must include a 'global_step' column.")
        sys.exit(1)

    # Identify episodic return column (excluding __MIN and __MAX)
    episodic_cols = [col for col in df.columns 
                     if "charts/episodic_return" in col and "__" not in col]

    if not episodic_cols:
        print("No episodic return column found.")
        sys.exit(1)

    episodic_col = episodic_cols[0]  # Assuming only one main column without __MIN/__MAX

    # Init W&B run
    wandb.init(project=args.project, entity=args.entity, name="csv_import")

    # Log data
    for _, row in df.iterrows():
        wandb.log({
            "global_step": row["global_step"],
            "charts/episodic_return": row[episodic_col]
        })

    print("Data successfully logged to W&B.")
    wandb.finish()

if __name__ == "__main__":
    main()