import argparse
import glob
import json
import matplotlib.pyplot as plt



def rank_models(result_files, metric="val_mse"):
    results = []
    
    for model_name in result_files:
        file = result_files[model_name]
        with open(file, "r") as f:
            data = json.load(f)
        
        name = data["config"]["name"]
        metric_values = data["history"][metric]

        best_value = min(metric_values)
        results.append((name, best_value, data))

    results.sort(key=lambda x: x[1])

    _METRIC_LABELS = {"val_mse": "MSE", "val_rmse": "RMSE", "val_mae": "MAE"}
    display_metric = _METRIC_LABELS.get(metric, metric)

    print(f"Best validation {display_metric} per model (sorted): \n")
    for i, (name, value, _) in enumerate(results,1):
        print(f"{i}. {name:30s} best {display_metric} = {value:.6f}")

    
    best_model_name, best_score, best_model_data = results[0]
    print("\nOverall best model:")
    print(f"{best_model_name} with best validation {display_metric} = {best_score:.6}")

    return best_model_name, best_score, best_model_data



def plot_model_losses(best_model_data):
    best_model_name = best_model_data["config"]["name"]
    history = best_model_data["history"]
    epochs = range(1, len(history["val_mse"]) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, history["train_mse"], label='Train MSE', marker='o', markersize=3)
    plt.plot(epochs, history["val_mse"], label='Validation MSE', marker='o', markersize=3)
    plt.plot(epochs, history["train_rmse"], label='Train RMSE', linestyle='--')
    plt.plot(epochs, history["val_rmse"], label='Validation RMSE', linestyle='--')
    plt.plot(epochs, history["train_mae"], label='Train MAE', linestyle=':')
    plt.plot(epochs, history["val_mae"], label='Validation MAE', linestyle=':')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training & Validation Losses for {best_model_name} (filtered)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank trained models by validation metric.")
    parser.add_argument("--dir",    default="checkpoints_3gw", help="Checkpoints directory to scan (default: checkpoints_3gw).")
    parser.add_argument("--metric", default="val_rmse",        help="Metric to rank by (default: val_rmse). Options: val_rmse, val_mse, val_mae.")
    args = parser.parse_args()

    files = glob.glob(f"{args.dir}/*_results*.json")
    if not files:
        print(f"No result JSONs found in {args.dir}/")
    else:
        result_files = {
            f.split("/")[-1].split("_results")[0]: f
            for f in files
        }
        rank_models(result_files, metric=args.metric)
