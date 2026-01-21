# Data prep and CLI orchestrator for Music Mood Clustering
import pandas as pd
import argparse
import json
import os
import shutil
import joblib
from run_clustering import run_clustering

try:
    df1 = pd.read_csv("dataset_A.csv")
    df2 = pd.read_csv("dataset_B.csv")
    print(f"Files loaded: A={len(df1)} rows, B={len(df2)} rows")
except FileNotFoundError:
    print("Files not found: dataset_A.csv and dataset_B.csv")
    exit()

parser = argparse.ArgumentParser()
expected_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
for col in expected_columns:
    if col not in df1.columns or col not in df2.columns:
        raise ValueError(f"Missing column: {col}")

df = pd.concat([df1, df2], ignore_index=True)
df = df.dropna(subset=expected_columns)
for col in expected_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=expected_columns)

# Subparsers for scenarios
subparsers = parser.add_subparsers(dest='command', help='Commands')

# 1. Cluster (single/multiple runs)
cluster_parser = subparsers.add_parser('cluster', help='Run clustering (single or multiple configs)')
cluster_parser.add_argument("--clusters", type=int, default=None, help="Fixed k for single run")
cluster_parser.add_argument("--config_file", type=str, default=None, help="JSON with list of configs for multiple")
cluster_parser.add_argument("--runs_dir", type=str, default="runs", help="Runs root dir")

# 2. Save snapshot
save_parser = subparsers.add_parser('save_snapshot', help='Save run as named snapshot')
save_parser.add_argument("run_id", type=str, help="Run ID to snapshot")
save_parser.add_argument("name", type=str, help="Snapshot name (e.g., Mood_v1)")
save_parser.add_argument("--runs_dir", type=str, default="runs")
save_parser.add_argument("--snapshots_dir", type=str, default="snapshots")

# 3. Load snapshot (predict)
load_parser = subparsers.add_parser('load_snapshot', help='Load snapshot and predict on new data')
load_parser.add_argument("name", type=str, help="Snapshot name")
load_parser.add_argument("--predict_csv", type=str, default=None, help="New CSV to predict on (optional)")
load_parser.add_argument("--output_dir", type=str, default="active_run", help="Target dir")
load_parser.add_argument("--snapshots_dir", type=str, default="snapshots")

# 4. Compare runs
compare_parser = subparsers.add_parser('compare', help='Compare two runs')
compare_parser.add_argument("run_id_a", type=str, help="First run ID")
compare_parser.add_argument("run_id_b", type=str, help="Second run ID")
compare_parser.add_argument("--runs_dir", type=str, default="runs")

args = parser.parse_args()

if args.command == 'cluster':
    # Multiple or single via run_clustering
    configs = []
    if args.config_file:
        with open(args.config_file, 'r') as f:
            configs = json.load(f)  # List of dicts
    elif args.clusters is not None:
        configs = [{
            'preprocessing': 'StandardScaler',
            'model': {'n_clusters': args.clusters, 'random_state': 42, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
            'features_used': ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        }]
    else:
        print("Error: Specify --clusters or --config_file")
        exit()

    all_metadata = []
    for cfg in configs:
        meta = run_clustering(df, df1, df2, cfg, args.runs_dir)
        all_metadata.append(meta)

    # Best selection (after cycle)
    if len(all_metadata) > 1:
        def composite_score(m):
            metrics = m['metrics']
            return (metrics['silhouette_score'] * metrics['calinski_harabasz_score'] /
                    (1 + metrics['davies_bouldin_score']))
        best_meta = max(all_metadata, key=composite_score)
        summary = {
            'best_run_id': best_meta['run_id'],
            'composite_score': composite_score(best_meta),
            'rule': 'max(silhouette * CH / (1 + DB))',
            'all_runs': {m['run_id']: m['metrics'] for m in all_metadata}
        }
        summary_path = os.path.join(args.runs_dir, 'best_run_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Best run: {best_meta['run_id']} (score: {composite_score(best_meta):.4f})")
        print(f"Summary saved: {summary_path}")

elif args.command == 'save_snapshot':
    src_dir = os.path.join(args.runs_dir, args.run_id)
    if not os.path.exists(src_dir):
        raise ValueError(f"Run {args.run_id} not found")
    dst_dir = os.path.join(args.snapshots_dir, args.name)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    print(f"Snapshot '{args.name}' saved: {dst_dir}")

elif args.command == 'load_snapshot':
    src_dir = os.path.join(args.snapshots_dir, args.name)
    if not os.path.exists(src_dir):
        raise ValueError(f"Snapshot '{args.name}' not found")
    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copy(os.path.join(src_dir, 'run_metadata.json'), os.path.join(args.output_dir, 'run_metadata.json'))
    shutil.copy(os.path.join(src_dir, 'model.joblib'), args.output_dir)

    if args.predict_csv:
        df_pred = pd.read_csv(args.predict_csv)
        features = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        X_pred = df_pred[features]

        kmeans = joblib.load(os.path.join(args.output_dir, 'model.joblib'))

        X_pred_scaled = X_pred
        df_pred['cluster'] = kmeans.predict(X_pred_scaled)

        pred_path = os.path.join(args.output_dir, 'predicted_clusters.csv')
        df_pred.to_csv(pred_path, index=False)
        print(f"Prediction saved: {pred_path}")

    print(f"Snapshot '{args.name}' loaded to {args.output_dir}")

elif args.command == 'compare':
    from compare_runs import compare_runs
    compare_runs(args.run_id_a, args.run_id_b, args.runs_dir)

else:
    parser.print_help()
