import streamlit as st
from models import *
import pandas as pd
import numpy as np


# Main Title
st.title("Machine Learning Studio")
# Sub Heading
st.subheader("Welcome to the Machine Learning Studio!")
# Intro Text
st.markdown('''Build. Train. Deploy. Machine Learning Made Easy.

Drag, drop, and build powerful ML models. Our intuitive studio empowers everyone to harness the power of data.  Start your journey today!''')

#Problem settings
st.sidebar.subheader("Setup your Problem Here")
# Problem Type Selection
problem_type = st.sidebar.selectbox("Pick your Problem Type", ["Regression", "Classification", "Clustering"])
# Taking the dataset file
dataset_file = st.sidebar.file_uploader("Upload your Dataset")
if problem_type != "Clustering":
    # Target Value
    target_y = st.sidebar.text_input("Enter the Target Variable Column Name")


if st.sidebar.button("Train"):
    st.subheader("Your Dataset:-")
    df = pd.read_csv(dataset_file)
    st.dataframe(df)
    if problem_type != "Clustering":
        cols = list(df.columns.values)
        cols.pop(cols.index(target_y))
        df = df[cols+[target_y]]


    model = Models(dataset_file)
    model.clean_and_scale_dataset()