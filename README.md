## The general idea of the project
It is a tool for the automatic detection of "latent mood groups" in music through unsupervised learning. The project analyzes the numerical characteristics of songs or listening sessions (audio + behavioral data), grouping them into clusters without predefined labels (such as "happy/sad"). The goal is to help with recommendations: e.g., "This session is like an 'energetic playlist' — suggest similar tracks." It is useful for music services (Spotify-like), where moods are not always explicit, but affect user engagement. Why unsupervised? Because moods are subjective — there are no "correct" labels, the model finds patterns on its own (like clustering stars in the sky).


## Features
This is a combination of audio and behavioral metrics (numeric features from CSV). It could probably be:

Feature_1: Total activity/tempo (e.g., BPM or average session speed).
Feature_2: Intensity/energy (e.g., volume or "emotional intensity" of a track).
Feature_3: Interactivity/valence (e.g., "gaiety" or user engagement).
Feature_4: Transitions/dynamics (e.g., skips or the frequency of track changes in a session).
Feature_5: Repeatability/patterns (e.g., full listens or searches/repeats of a song).

This is behavioral (likes, swipes, listening to the end, searches) + audio (from librosa/Spotify API). Two datasets (A/B) — for variety: e.g., A=classical (calm, low feature_1/2), B=rock (high, noisy) — concat enhances variation, the model sees a "wide range of moods".


## Clusters
Clusters are "mood portraits": groups of sessions/songs with similar patterns (not labels, but emergent characteristics). e.g.:

Cluster 0: High feature_2/4, low 3 — "energetic, dynamic" (dance/rock, lots of skips, but full listens).
Cluster 1: Low feature_1/5, high 3 — "calm, thoughtful" (ballads/classics, repetitions, few transitions).

This is latent: the model finds the "hidden" (e.g., "light thoughtful" — implicitly from the combo features).

Why A/B datasets? For augmentation: A is one genre/user-type (e.g., classical, clean), B is another (rock, behavioral noisy). Concat = 500+ rows, the model sees diversity (without trivial clusters).

Metrics (silhouette/CH/DB/inertia)? The quality is evaluated by: Silhouette — "clarity of groups" (>0.5=good); CH — "structure" (higher=strong patterns); DB — "intersections" (lower=better); Inertia — "compactness" (lower=tight moods). Composite for best: balance.

PCA-plot? Reduction 5D → 2D (projection: PC1="energy-axis", PC2="mood-axis") — visual clusters (clusters of dots = moods).

Application: Recommendations (cluster → playlist), A/B tests (compare runs), inference (load snapshot → predict on new sessions). Scale: From toy (60 lines) to prod (MLflow for logging).

The project is an unsupervised explorer: It finds "invisible connections" in music as a psychological profile of a playlist.

# How to run
## For Web UI
To run the web interface, make sure the streamlit is installed and run
```
streamlit run app.py
```
## For CLI
Run in the terminal:
```
python datacheck.py cluster --clusters 4
```

After that you can check /runs/ directory
clustered_output.csv: Your songs + the 'cluster' column (0=quiet, 1=energetic or something like that).
clusters_plot.png
run_metadata.json: "Passport" — config, metrics (silhouette=0.55 — "are groups good?"), file paths.
model.joblib — for later

python cluster_feature_profiles.py --input_csv runs/{run_id}/clustered_output.csv

You can create configs.json for multiple run like this:
```
[
  {
    "preprocessing": "StandardScaler",
    "model": {"n_clusters": 3, "init": "k-means++", "n_init": 10, "max_iter": 300, "random_state": 42},
    "features_used": ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
  },
  {
    "preprocessing": "MinMaxScaler",
    "model": {"n_clusters": null},  // auto-k!
    "features_used": ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
  },
  {
    "preprocessing": "RobustScaler",
    "model": {"n_clusters": 5, "random_state": 123},
    "features_used": ["feature_1", "feature_2"]  // Только 2 фичи для теста
  }
]
```

And then do follow:
```
python datacheck.py cluster --config_file configs.json
```

To compare:
```
python compare_runs.py run_id_1 run_id_2
```

After first or any another run you can save it's snapshot:
```
python datacheck.py save_snapshot {run_id}} Mood_v4
```

So here predict on new dataset without learning: 
```
python datacheck.py load_snapshot Mood_v1 --predict_csv new_data.csv
```

And this:
```
python cluster_feature_profiles.py --input_csv active_run/predicted_clusters.csv
```
