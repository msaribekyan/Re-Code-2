import json
import os
import hashlib
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def run_clustering(df, df1, df2, config, runs_dir="runs"):
    # 1. Generate unique run_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    k_str = str(config['model']['n_clusters']) if config['model']['n_clusters'] is not None else 'auto'
    run_id = f"{timestamp}_{config_hash}_{config['preprocessing']}_{k_str}"
    output_dir = os.path.join(runs_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nRunning config for {run_id}: {config}")

    # Features
    features = config.get('features_used', ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'])
    X = df[features].copy()

    # 2. Preprocessing: Select and apply scaler
    scaler_type = config['preprocessing']
    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaler_type == 'RobustScaler':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unsupported scaler: {scaler_type}")

    X_scaled = scaler.fit_transform(X)

    # 3. Determine n_clusters (auto if None)
    model_params = config['model']
    random_state = model_params.get('random_state', 42)
    if model_params['n_clusters'] is None:
        best_score = -1
        best_k = 2
        for k in range(2, 10):
            kmeans_temp = KMeans(
                n_clusters=k,
                init=model_params.get('init', 'k-means++'),
                n_init=model_params.get('n_init', 10),
                max_iter=model_params.get('max_iter', 300),
                random_state=random_state
            )
            labels_temp = kmeans_temp.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels_temp)
            if score > best_score:
                best_score = score
                best_k = k
        n_clusters = best_k
    else:
        n_clusters = model_params['n_clusters']

    # 4. Train KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        init=model_params.get('init', 'k-means++'),
        n_init=model_params.get('n_init', 10),
        max_iter=model_params.get('max_iter', 300),
        random_state=random_state
    )
    cluster_labels = kmeans.fit_predict(X_scaled)
    df_copy = df.copy()
    df_copy['cluster'] = cluster_labels

    # 5. Compute metrics
    silhouette = silhouette_score(X_scaled, cluster_labels)
    ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
    db_score = davies_bouldin_score(X_scaled, cluster_labels)
    inertia = kmeans.inertia_
    metrics = {
        'silhouette_score': silhouette,
        'calinski_harabasz_score': ch_score,
        'davies_bouldin_score': db_score,
        'inertia': inertia,
        'n_clusters': n_clusters
    }

    # 6. PCA and Plot
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    for c in range(n_clusters):
        plt.scatter(X_2d[cluster_labels == c, 0], X_2d[cluster_labels == c, 1], label=f"Cluster {c}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"Clusters visualization (run: {run_id})")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "clusters_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    pca_df = pd.DataFrame({
        "PC1": X_2d[:, 0],
        "PC2": X_2d[:, 1],
        "cluster": cluster_labels
    })
    pca_df_path = os.path.join(output_dir, "pca_df.csv")
    pca_df.to_csv(pca_df_path, index=False)   
 
    # 7. Save everything
    # CSV
    csv_path = os.path.join(output_dir, "clustered_output.csv")
    df_copy.to_csv(csv_path, index=False)

    # Model snapshot (joblib for sklearn)
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(kmeans, model_path)

    # Copies of datasets
    df1.to_csv(os.path.join(output_dir, "dataset_A.csv"), index=False)
    df2.to_csv(os.path.join(output_dir, "dataset_B.csv"), index=False)

    # Metadata JSON
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "preprocessing": scaler_type,
            "model": {
                "name": "KMeans",
                "n_clusters": n_clusters,  # Use final n_clusters (after auto)
                "init": model_params.get('init', 'k-means++'),
                "n_init": model_params.get('n_init', 10),
                "max_iter": model_params.get('max_iter', 300),
                "random_state": random_state
            },
            "features_used": features
        },
        "metrics": metrics,
        "file_paths": {
            "output_csv": "clustered_output.csv",  # Basename only
            "plot": "clusters_plot.png",
            "model_snapshot": "model.joblib"
        }
    }

    metadata_path = os.path.join(output_dir, "run_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Results saved to {output_dir}")
    print(f"Silhouette score: {silhouette:.4f}")

    return metadata
