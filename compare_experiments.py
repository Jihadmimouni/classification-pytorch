import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

def compare_experiments():
    experiment_names = [
        "exp_baseline_adam",
        "exp_optimizer_sgd",
        "exp_lr_high",
        "exp_augmentation"
    ]
    
    all_runs = []
    
    print(f"{'Experiment':<25} | {'Run Name':<30} | {'Avg Val Acc':<15} | {'Avg Val Loss':<15}")
    print("-" * 95)
    
    for exp_name in experiment_names:
        experiment = mlflow.get_experiment_by_name(exp_name)
        if experiment is None:
            print(f"Experiment {exp_name} not found.")
            continue
            
        # Search for runs in this experiment
        # We are looking for the parent runs (cross-validation summaries)
        # They usually have metrics like 'cv_avg_best_val_accuracy'
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="metrics.cv_avg_best_val_accuracy > 0",
            order_by=["attribute.start_time DESC"]
        )
        
        if runs.empty:
            print(f"{exp_name:<25} | {'No runs found':<30} | {'N/A':<15} | {'N/A':<15}")
            continue
            
        # Get the latest run
        latest_run = runs.iloc[0]
        
        run_name = latest_run['tags.mlflow.runName']
        val_acc = latest_run['metrics.cv_avg_best_val_accuracy']
        val_loss = latest_run['metrics.cv_avg_best_val_loss']
        
        print(f"{exp_name:<25} | {run_name:<30} | {val_acc:<15.4f} | {val_loss:<15.4f}")
        
        all_runs.append({
            "Experiment": exp_name,
            "Run Name": run_name,
            "Accuracy": val_acc,
            "Loss": val_loss
        })

    print("-" * 95)
    
    # Optional: Save to CSV
    if all_runs:
        df = pd.DataFrame(all_runs)
        df.to_csv("experiment_comparison.csv", index=False)
        print("\nComparison saved to experiment_comparison.csv")

if __name__ == "__main__":
    compare_experiments()
