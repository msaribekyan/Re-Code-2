import streamlit as st
import re
import os

datasets = ["text.csv", "1.csv"]

def upload(filename, file):
    if not filename:
        st.write("Empty dataset name")
        return None
    
    if not re.fullmatch(r"[A-Za-z0-9_]+", filename):
        st.write("Filename can include only letters, numbers and underscore")
        return None

    if (filename+ ".csv") in datasets:
        st.write("Dataset already exists")
        return None

    if file is None:
        st.write("File not uploaded")
        return None

    with open(os.path.join("datasets", filename + ".csv", ),"wb") as f:
        f.write(file.getbuffer())

st.title("Re-Code: Cluster")

st.header("New run")
st.write("Select pre-processing and model parameters, upload or select the input dataset and cluster the data.")

scaler_method = st.selectbox("Scaler", ["StandardScaler", "MinMaxScaler", "RobustScaler"])
n_clusters = st.slider("Number of clusters", min_value=2, max_value=10)
random_state = st.slider("Random state", value=42, min_value=0, max_value=100)
dataset_method = st.selectbox("Dataset", ["Select dataset from list", "Upload dataset"])

if dataset_method == "Upload dataset":
    filename = st.text_input("Dataset name")
    file = st.file_uploader("Upload dataset",type=["csv"])
elif dataset_method == "Select dataset from list":
    dataset = st.selectbox("Select dataset", datasets)

if st.button("Run", key="button_run"):
    if dataset_method == "Upload dataset":
        dataset = upload(filename, file)
    if dataset is not None:    
        st.write("Using the dataset: " + dataset)
    else:
        st.write("Error")


