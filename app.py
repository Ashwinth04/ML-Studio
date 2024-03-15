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
problem_type = st.sidebar.selectbox("Pick your Problem Type", ["Regression", "Classification", "Clustering"])\

if problem_type == "Regression":
    state = 1
elif problem_type == "Classification":
    state = 2
elif problem_type == "Clustering":
    state = 3
else:
    state = 4

dataset_file = st.sidebar.file_uploader("Upload your Dataset")
st.subheader("Your Dataset:-")
df = pd.read_csv(dataset_file)
st.dataframe(df)
if state == 1 or state == 2:
    target_y = st.sidebar.text_input("Enter the Target Variable Column Name")


if st.sidebar.button("Train"):
    if (state == 1 or state == 2) and target_y != "":
        cols = list(df.columns.values)
        cols.pop(cols.index(target_y))
        df = df[cols+[target_y]]

    model = Models(df)
    model.clean_and_scale_dataset()

    if state == 1:
        regression_models = ["Linear Regression", "Decision Tree Regression", "SVR", "Ridge Regression", "Lasso Regression",
                             "ElasticNet", "Random Forest Regressor", "Multi-Layer Perceptron", "KNN Regressor", 
                             "Gradient Boosting Regressor"]
        metrics = ["R2-Score"]
        lr = list(model.linear_regression())
        dtr = list(model.dtree_regressor())
        svr = list(model.SVR())
        rr = list(model.ridge_regression())
        lsr = list(model.lasso())
        en = list(model.elasticnet())
        rfr = list(model.random_forest_regressor())
        mlpr = list(model.mlp_regressor())
        knnr = list(model.knn_regressor())
        gbr = list(model.gradient_boost_regressor())

        regression_funcs = {
            "Linear Regression": lr[0],
            "Decision Tree Regression": dtr[0],
            "SVR": svr[0],
            "Ridge Regression": rr[0],
            "Lasso Regression": lsr[0],
            "ElasticNet": en[0],
            "Random Forest Regressor": rfr[0],
            "Multi-Layer Perceptron": mlpr[0],
            "KNN Regressor": knnr[0],
            "Gradient Boosting Regressor": gbr[0]
        }

        regressors_table = {
            "Linear Regression": lr[1:],
            "Decision Tree Regression": dtr[1:],
            "SVR": svr[1:],
            "Ridge Regression": rr[1:],
            "Lasso Regression": lsr[1:],
            "ElasticNet": en[1:],
            "Random Forest Regressor": rfr[1:],
            "Multi-Layer Perceptron": mlpr[1:],
            "KNN Regressor": knnr[1:],
            "Gradient Boosting Regressor": gbr[1:]
        }
        comp_table = pd.DataFrame.from_dict(regressors_table)
        comp_table.index = metrics
        comp_table = comp_table.transpose()

    elif state == 2:
        classification_models = ["SVM", "Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", 
                                 "AdaBoost", "Multi-Layer Perceptron", "Gradient Boosting", "Random Forest"]
        metrics = ["Accuracy"]
        s = list(model.SVM())
        lor = list(model.logistic_regression())
        dt = list(model.dtree_classifier())
        knn = list(model.knn_classifier())
        nb = list(model.naivebayes())
        ab = list(model.adaboost())
        mlp = list(model.mlp())
        gb = list(model.gradient_boost())
        rf = list(model.random_forest_classifier())

        classification_funcs = {
            "SVM": s[0],
            "Logistic Regression": lor[0],
            "Decision Tree": dt[0],
            "KNN": knn[0],
            "Naive Bayes": nb[0],
            "AdaBoost": ab[0],
            "Multi-Layer Perceptron": mlp[0],
            "Gradient Boosting": gb[0],
            "Random Forest": rf[0]
        }

        classifiers_table = {
            "SVM": s[1:],
            "Logistic Regression": lor[1:],
            "Decision Tree": dt[1:],
            "KNN": knn[1:],
            "Naive Bayes": nb[1:],
            "AdaBoost": ab[1:],
            "Multi-Layer Perceptron": mlp[1:],
            "Gradient Boosting": gb[1:],
            "Random Forest": rf[1:]
        }
        comp_table = pd.DataFrame.from_dict(classifiers_table)
        comp_table.index = metrics
        comp_table = comp_table.transpose()
    elif state == 3:
        pass
    else:
        pass

    st.subheader("Comparision Table")
    st.dataframe(comp_table)

    st.sidebar.subheader("Test your Model Here")
    if state == 1:
        model_name = st.sidebar.selectbox("Pick your Model", ["Best Model"] + regression_models)
    elif state == 2:
        model_name = st.sidebar.selectbox("Pick your Model", ["Best Model"] + classification_models)
    elif state == 3:
        pass
    else:
        pass
    
    