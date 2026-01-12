import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import re
import os
import json
import plotly.express as px
from run_clustering import run_clustering

if not os.path.exists("datasets"):
    os.makedirs("datasets")
datasets = os.listdir("datasets")

if not os.path.exists("runs"):
    os.makedirs("runs")
runs = os.listdir("runs")

def upload(filename, file, file_number):
    if not filename:
        st.write(f"Empty dataset {file_number} name")
        return None
    
    if not re.fullmatch(r"[A-Za-z0-9_]+", filename):
        st.write(f"Filename {file_number}can include only letters, numbers and underscore")
        return None

    if (filename+ ".csv") in datasets:
        st.write(f"Dataset {file_number} already exists")
        return None

    if file is None:
        st.write(f"File {file_number} not uploaded")
        return None
    df = pd.read_csv(file)

    expected_clumns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    for col in expected_clumns:
        if col not in df.columns:
            st.write(f"There is no column: {col} in dataset {file_number}")
            return None

    df = df.dropna(subset=expected_clumns)
    df.to_csv(os.path.join("datasets", filename + ".csv"), encoding='utf-8', index=False)
    datasets.append(filename + ".csv")

    return filename + ".csv"

def show_run(run_id, key):
        with open(os.path.join("runs", run_id, "run_metadata.json"), 'r') as f:
            metadata = json.load(f)

        pca_df = pd.read_csv(os.path.join("runs", run_id, "pca_df.csv"))

        fig = px.scatter(
            x=pca_df["PC1"],
            y=pca_df["PC2"],
            color="Cluster " + pca_df["cluster"].astype(str),
            title='K-Means Clusters',
            labels={'x':'PCA1', 'y':'PCA2', 'color':'Cluster'}
        )

        st.write("Preprocessing: " + metadata["config"]["preprocessing"])
        st.write("Model: " + metadata["config"]["model"]["name"])
        st.write("Clusters: " + str(metadata["config"]["model"]["n_clusters"]))
        st.write("Random State: " + str(metadata["config"]["model"]["random_state"]))
        st.plotly_chart(fig, key="fig"+key)

        st.write("Silhouette score: " + str(metadata["metrics"]["silhouette_score"]))
        st.write("Calinski Harabasz score: " + str(metadata["metrics"]["calinski_harabasz_score"]))
        st.write("Davies Bouldin score: " + str(metadata["metrics"]["davies_bouldin_score"]))
        st.write("Inertia: " + str(metadata["metrics"]["inertia"]))

        st.write("Download files")
        with open(os.path.join("runs", run_id, "clustered_output.csv"), "rb") as file:
            st.download_button(label="Download clustered output",
                               data=file,
                               file_name="clustered_output.csv",
                               mime="text/csv",
                               key="dow1"+key
            )
        with open(os.path.join("runs", run_id, "clusters_plot.png"), "rb") as file:
            st.download_button(label="Download cluster plot",
                               data=file,
                               file_name="clusters_plot.png",
                               mime="image/png",
                               key="dow2"+key
            )

def page_new_run():
    st.set_page_config(layout="centered")
    st.header("New run")
    st.write("Select pre-processing and model parameters, upload or select the input dataset and cluster the data.")

    scaler_method = st.selectbox("Scaler", ["StandardScaler", "MinMaxScaler", "RobustScaler"])
    n_clusters = st.slider("Number of clusters", min_value=2, max_value=10)
    random_state = st.slider("Random state", value=42, min_value=0, max_value=100)
    st.write("Select which features to use for model training")
    dataset_method = st.selectbox("Dataset", ["Select dataset from list", "Upload dataset"])

    if dataset_method == "Upload dataset":
        filename_A = st.text_input("Dataset A name")
        file_A = st.file_uploader("Upload dataset A",type=["csv"])
        filename_B = st.text_input("Dataset B name")
        file_B = st.file_uploader("Upload dataset B",type=["csv"])
    elif dataset_method == "Select dataset from list":
        dataset_A = st.selectbox("Select dataset A", datasets)
        dataset_B = st.selectbox("Select dataset B", datasets)

    if st.button("Run", key="button_run"):
        if dataset_method == "Upload dataset":
            dataset_A = upload(filename_A, file_A, "A")
            dataset_B = upload(filename_B, file_B, "B")
        if dataset_A is None or dataset_B is None:
            st.write("Dataset selection error")
        elif dataset_A == dataset_B:
            st.write("Selected datasets can not be the same")
        else:
            df1 = pd.read_csv(os.path.join("datasets", dataset_A))
            df2 = pd.read_csv(os.path.join("datasets", dataset_B))
        
            df = pd.concat([df1, df2], ignore_index=True)
            config = {
                "preprocessing": scaler_method,
                "model": {
                    "n_clusters": n_clusters,
                    "random_state": random_state,
                },
            }
            metadata = run_clustering(df, df1, df2, config)
            run_id = metadata["run_id"]

            show_run(run_id, "calc")

def view_run():
    st.set_page_config(layout="centered")
    st.header("View run")
    st.write("Select run to get plot and metrics")

    run_id = st.selectbox("Run", runs)
    if run_id is not None:
        show_run(run_id, "view")

def compare_runs():
    st.set_page_config(layout="wide")
    st.header("Compare runs")
    num_comp = st.slider("Number of runs to compare", min_value=2, max_value=4)
    cols = st.columns(num_comp)
    for i, col in enumerate(cols):
        with col:
            st.write("Select run to get plot and metrics")
            
            run_id = st.selectbox("Run", runs, key="sel" + str(i))
            if run_id is not None:
                runs.remove(run_id)
                show_run(run_id, "run" + str(i))

st.title("Re-Code: Cluster")

page = st.selectbox("Page", ["New run", "View run", "Compare runs"])

if page == "New run":
    page_new_run()
elif page == "View run":
    view_run()
elif page == "Compare runs":
    compare_runs()
