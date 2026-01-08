# one run function
def run_clustering(config, run_id, output_dir):
    print(f"\nRunning config for {run_id}: {config}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Поддержка разных preprocessing (пока базовая: scaler_type, но можно расширить)
    if config.get('preprocessing', {}).get('scaler_type', 'standard') == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

    if config.get('n_clusters') is not None:
        n_clusters = config['n_clusters']
    else:
        best_score = -1
        best_k = 2
        for k in range(2, 10):
            kmeans = KMeans(n_clusters=k, random_state=config.get('random_state', 42))
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score = score
                best_k = k
        n_clusters = best_k

    kmeans = KMeans(n_clusters=n_clusters, random_state=config.get('random_state', 42))
    cluster_labels = kmeans.fit_predict(X_scaled)
    df_copy = df.copy()
    df_copy['cluster'] = cluster_labels

    # Метрики
    silhouette = silhouette_score(X_scaled, cluster_labels)
    inertia = kmeans.inertia_
    metrics = {'silhouette_score': silhouette, 'inertia': inertia, 'n_clusters': n_clusters}

    ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
    db_score = davies_bouldin_score(X_scaled, cluster_labels)

    # Сохранения
    os.makedirs(output_dir, exist_ok=True)

    # CSV с кластерами
    csv_path = os.path.join(output_dir, "clustered_output.csv")
    df_copy.to_csv(csv_path, index=False)

    # Plot
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

    # Config и metrics
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Копии датасетов
    df1.to_csv(os.path.join(output_dir, "dataset_A.csv"), index=False)
    df2.to_csv(os.path.join(output_dir, "dataset_B.csv"), index=False)

    print(f"Results saved to {output_dir}")
    print(f"Silhouette score: {silhouette:.4f}")


    return metrics


# Получение configs
configs = []
if args.config_file:
    with open(args.config_file, 'r') as f:
        configs = json.load(f)
elif args.clusters is not None:
    configs = [{'n_clusters': args.clusters, 'random_state': 42}]
else:
    # Fallback: auto-detect single
    configs = [{'random_state': 42}]

# Создание runs dir
os.makedirs(args.runs_dir, exist_ok=True)

# Цикл по configs
all_metrics = {}
for i, config in enumerate(configs):
    # Уникальный ID: timestamp + hash config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    run_id = f"{timestamp}_{config_hash}"
    output_dir = os.path.join(args.runs_dir, run_id)

    metrics = run_clustering(config, run_id, output_dir)
    all_metrics[run_id] = metrics

# Summary всех runs
print("\n=== Summary of all runs ===")
for run_id, m in all_metrics.items():
    print(f"{run_id}: silhouette={m['silhouette_score']:.4f}, clusters={m['n_clusters']}")
