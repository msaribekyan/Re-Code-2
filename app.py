import streamlit as st
import pandas as pd
import re
import os
from run_clustering import run_clustering

datasets = os.listdir("datasets")

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


st.title("Re-Code: Cluster")

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
        run_id = run_clustering(df, df1, df2, config)


