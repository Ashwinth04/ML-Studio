import streamlit as st
from models import *
import pandas as pd
import numpy as np
# import shutil
# from nn import *


# Main
st.title("Machine Learning Studio")
st.subheader("Welcome to the Machine Learning Studio!")
st.markdown('''Build. Train. Deploy. Machine Learning Made Easy.

Drag, drop, and build powerful ML models. Our intuitive studio empowers everyone to harness the power of data.  Start your journey today!''')

# Side bar
st.sidebar.subheader("Setup your Problem Here")
problem_type = st.sidebar.selectbox("Pick your Problem Type", ["Regression", "Classification", "Clustering", "Image Classification"])

if problem_type == "Regression":
    state = 1
elif problem_type == "Classification":
    state = 2
elif problem_type == "Clustering":
    state = 3
else:
    state = 4

if state == 4:
    img_zip_file = st.sidebar.file_uploader("Upload your Dataset", type=['zip'])
    print(img_zip_file.name)
else:
    dataset_file = st.sidebar.file_uploader("Upload your Dataset", type=['csv'])
if state != 4:
    if dataset_file:
        st.subheader("Your Dataset:-")
        df = pd.read_csv(dataset_file)
        st.dataframe(df)
        if state == 1 or state == 2:
            target_y = st.sidebar.text_input("Enter the Target Variable Column Name (Leave Blank to use Last Column)")


if st.sidebar.button("Train"):

    comp_table_flag = 1

    if state != 4:
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

            metrics = list(lr[1].keys())

            regressors_table = {
                "Linear Regression": list(lr[1].values()),
                "Decision Tree Regression": list(dtr[1].values()),
                "SVR": list(svr[1].values()),
                "Ridge Regression": list(rr[1].values()),
                "Lasso Regression": list(lsr[1].values()),
                "ElasticNet": list(en[1].values()),
                "Random Forest Regressor": list(rfr[1].values()),
                "Multi-Layer Perceptron": list(mlpr[1].values()),
                "KNN Regressor": list(knnr[1].values()),
                "Gradient Boosting Regressor": list(gbr[1].values())
            }
            comp_table = pd.DataFrame.from_dict(regressors_table)
            comp_table.index = metrics
            comp_table = comp_table.transpose()

        elif state == 2:
            classification_models = ["SVM", "Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", 
                                    "AdaBoost", "Multi-Layer Perceptron", "Gradient Boosting", "Random Forest"]
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

            metrics = list(s[1].keys())

            classifiers_table = {
                "SVM": list(s[1].values()),
                "Logistic Regression": list(lor[1].values()),
                "Decision Tree": list(dt[1].values()),
                "KNN": list(knn[1].values()),
                "Naive Bayes": list(nb[1].values()),
                "AdaBoost": list(ab[1].values()),
                "Multi-Layer Perceptron": list(mlp[1].values()),
                "Gradient Boosting": list(gb[1].values()),
                "Random Forest": list(rf[1].values())
            }
            comp_table = pd.DataFrame.from_dict(classifiers_table)
            comp_table.index = metrics
            comp_table = comp_table.transpose()
        else:
            comp_table_flag = 0
            clustering_models = ["K-Means Clustering", "Agglomerative Clustering", "Density-Based Clustering"]
            num_cluster = st.sidebar.text_input("Enter the Number of Expected Clusters (Leave blank for letting us decide)")
            eps = st.sidebar.text_input("Radius of the Density (Leave Blank for 0.5)")
            min_samples = st.sidebar.text_input("Minimum Number of Samples around the Radius (Leave Blank for 5)")

            if st.sidebar.button("Set Parameters"):
                if num_cluster != "":
                    kc = list(model.kmeans(int(num_cluster)))
                    ac = list(model.agglomerative_clustering(int(num_cluster)))
                else:
                    kc = list(model.kmeans(-1))
                    ac = list(model.agglomerative_clustering(-1))
                if eps == "":
                    if min_samples == "":
                        dc = list(model.dbscan())
                    else:
                        dc = list(model.dbscan(min_samples=min_samples))
                else:
                    if min_samples == "":
                        dc = list(model.dbscan(eps=eps))
                    else:
                        dc = list(model.dbscan(eps=eps, min_samples=min_samples))

                clustering_funcs = {
                    "K-Means Clustering": kc[0],
                    "Agglomerative Clustering": ac[0],
                    "Density-Based Clustering": dc[0]
                }

                metrics = list(kc[2].keys())

                clustering_table = {
                    "K-Means Clustering": list(kc[2].values()),
                    "Agglomerative Clustering": list(ac[2].values()),
                    "Density-Based Clustering": list(dc[2].values())
                }
                comp_table = pd.DataFrame.from_dict(clustering_table)
                comp_table.index = metrics
                comp_table = comp_table.transpose()
                comp_table_flag = 1
    
    else:
        # comp_table_flag = 0
        # shutil.unpack_archive(img_zip_file.name)
        # dir_name = img_zip_file.name.split(".")[0]
        # nn = NeuralNetwork(dir_name)
        # nn.make_train_val_dirs()
        # nn.create_dataset()
        # nn.train_val_gens()
        # nn.model()
        pass

    if comp_table_flag:
        st.subheader("Comparision Table")
        st.dataframe(comp_table)