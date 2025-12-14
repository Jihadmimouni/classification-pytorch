import subprocess
import os
import sys
import time

def run_experiment(name, params):
    print(f"Running experiment: {name}")
    print(f"Params: {params}")
    
    # Set environment variable for MLflow experiment name
    env = os.environ.copy()
    env["MLFLOW_EXPERIMENT_NAME"] = name
    
    cmd = [
        sys.executable, "main.py",
        "--mode", "train",
        "--data_path", "data/train",
        "--use_mlflow"
    ]
    
    if "batch_size" in params:
        cmd.extend(["--batch_size", str(params["batch_size"])])
    
    if "learning_rate" in params:
        cmd.extend(["--learning_rate", str(params["learning_rate"])])
        
    if "optimizer" in params:
        cmd.extend(["--optimizer", params["optimizer"]])
        
    if params.get("augment", False):
        cmd.append("--augment")
        
    # Run the command
    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"Experiment {name} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Experiment {name} failed with error: {e}\n")

def main():
    experiments = [
        {
            "name": "exp_baseline_adam",
            "params": {
                "optimizer": "adam",
                "learning_rate": 1e-4,
                "augment": False
            }
        },
        {
            "name": "exp_optimizer_sgd",
            "params": {
                "optimizer": "sgd",
                "learning_rate": 1e-4,
                "augment": False
            }
        },
        {
            "name": "exp_lr_high",
            "params": {
                "optimizer": "adam",
                "learning_rate": 1e-3,
                "augment": False
            }
        },
        {
            "name": "exp_augmentation",
            "params": {
                "optimizer": "adam",
                "learning_rate": 1e-4,
                "augment": True
            }
        }
    ]
    
    print(f"Starting {len(experiments)} experiments...")
    
    for exp in experiments:
        run_experiment(exp["name"], exp["params"])
        # Small delay between runs
        time.sleep(2)
        
    print("All experiments completed.")

if __name__ == "__main__":
    main()
