import json
import pandas as pd
import argparse
import os
from sklearn.metrics import adjusted_rand_score


def compare_runs(run_id_a, run_id_b, runs_dir="runs"):
    path_a = os.path.join(runs_dir, run_id_a)
    path_b = os.path.join(runs_dir, run_id_b)

    # 1. Load metadata
    with open(os.path.join(path_a, "run_metadata.json"), 'r') as f:
        meta_a = json.load(f)
    with open(os.path.join(path_b, "run_metadata.json"), 'r') as f:
        meta_b = json.load(f)

    print(f"=== Comparison: {meta_a['run_id']} vs {meta_b['run_id']} ===\n")

    # 2. Compare configurations (nested get for model)
    print("--- Configuration changes ---")
    cfg_keys = ['preprocessing', 'model.n_clusters']
    for key in cfg_keys:
        if key == 'preprocessing':
            val_a = meta_a['config'].get(key, 'N/A')
            val_b = meta_b['config'].get(key, 'N/A')
        elif key == 'model.n_clusters':
            val_a = meta_a['config']['model'].get('n_clusters', 'N/A')
            val_b = meta_b['config']['model'].get('n_clusters', 'N/A')
        status = "CHANGED" if val_a != val_b else "identically"
        print(f"{key}: {val_a} -> {val_b} ({status})")

    # 3. Compare metrics (DataFrame for Δ)
    print("\n--- The difference in metrics ---")
    metrics_df = pd.DataFrame({
        'Metric': list(meta_a['metrics'].keys()),
        f'{run_id_a}': [meta_a['metrics'][k] for k in meta_a['metrics']],
        f'{run_id_b}': [meta_b['metrics'].get(k, 0) for k in meta_a['metrics']],
        'Delta (abs)': [meta_b['metrics'].get(k, 0) - meta_a['metrics'][k] for k in meta_a['metrics']]
    })
    print(metrics_df.round(4).to_string(index=False))

    # 4. Compare assignments (ARI, with check sizes)
    # Basename join
    csv_a = os.path.join(path_a, "clustered_output.csv")
    csv_b = os.path.join(path_b, "clustered_output.csv")
    if not os.path.exists(csv_a) or not os.path.exists(csv_b):
        print("\n--- Clusters stability ---")
        print("⚠️ CSV not found — check paths.")
        return
    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(csv_b)
    print(f"Dataset sizes: A={len(df_a)} rows, B={len(df_b)} rows")
    if len(df_a) != len(df_b):
        print("⚠️ Different sizes — ARI doesn't count (compare metrics above).")
        ari_score = None
    else:
        ari_score = adjusted_rand_score(df_a['cluster'], df_b['cluster'])
        print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
        if ari_score > 0.9:
            print("Res: Clusters almost identical.")
        elif ari_score > 0.5:
            print("Res: Groups have changed moderately.")
        else:
            print("Res: The structure has been rebuilt.")

    # Save the report
    comparison = {
        'configs': [meta_a['config'], meta_b['config']],
        'metrics_df': metrics_df.to_dict('records'),
        'ari': ari_score,
        'sizes': {'a': len(df_a), 'b': len(df_b)}
    }
    comp_path = os.path.join(runs_dir, f"comparison_{run_id_a}_vs_{run_id_b}.json")
    with open(comp_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\nThe report is saved: {comp_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two clustering runs")
    parser.add_argument("run_id_a", type=str, help="First run ID")
    parser.add_argument("run_id_b", type=str, help="Second run ID")
    parser.add_argument("--runs_dir", type=str, default="runs", help="Runs root dir")
    args = parser.parse_args()
    compare_runs(args.run_id_a, args.run_id_b, args.runs_dir)