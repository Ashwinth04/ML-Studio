import streamlit as st
from temp_model import *
import pandas as pd
import numpy as np


# Main
st.title("Machine Learning Studio")
st.subheader("Welcome to the Machine Learning Studio!")
st.markdown('''Build. Train. Deploy. Machine Learning Made Easy.

Drag, drop, and build powerful ML models. Our intuitive studio empowers everyone to harness the power of data.  Start your journey today!''')

# Side bar
st.sidebar.subheader("Setup your Problem Here")
problem_type = st.sidebar.selectbox("Pick your Problem Type", ["Regression", "Classification", "Clustering"])
dataset_file = st.sidebar.file_uploader("Upload your Dataset")
if problem_type != "Clustering":
    target_y = st.sidebar.text_input("Enter the Target Variable Column Name")


if st.sidebar.button("Train"):
    st.subheader("Your Dataset:-")
    df = pd.read_csv(dataset_file)
    st.dataframe(df)
    if problem_type != "Clustering" and target_y != "":
        cols = list(df.columns.values)
        cols.pop(cols.index(target_y))
        df = df[cols+[target_y]]

    model = Models(df)
    model.clean_and_scale_dataset()

    if problem_type == "Regression":
        regression_models = ["Linear Regression", "DTree Regression", "SVR"]
        metrics = ["R2-Score"]
        lr = list(model.linear_regression())
        dtr = list(model.dtree_regressor())
        svr = list(model.SVR())

        regressors_table = {
            "Linear Regression": lr[1:],
            "Decision Tree Regression": dtr[1:],
            "SVR": svr[1:]
        }
        comp_table = pd.DataFrame.from_dict(regressors_table)
        comp_table.index = metrics

    elif problem_type == "Classification":
        regression_models = ["SVM", "Logistic Regression", "Decision Tree", "KNN"]
        metrics = ["Accuracy"]
        svm = list(model.SVM())
        lor = list(model.logistic_regression())
        dt = list(model.decision_tree())
        knn = list(model.knn())

        classifiers_table = {
            "SVM": svm[1:],
            "Logistic Regression": lor[1:],
            "Decision Tree": dt[1:],
            "KNN": knn[1:]
        }
        comp_table = pd.DataFrame.from_dict(classifiers_table)
        comp_table.index = metrics
    else:
        pass

    st.subheader("Comparision Table")
    st.dataframe(comp_table)