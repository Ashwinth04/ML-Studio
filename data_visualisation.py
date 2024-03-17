import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as shc
from models import *

# Function to provide suggestions for regression model based on R2 score
def threshold_regression(r2_score, threshold=0.7):
    if r2_score >= threshold:
        suggestion = f"The R2 score ({r2_score:.2f}) indicates a good fit for the regression model. No further action is required."
    else:
        if r2_score >= 0.6:
            suggestion = f"The R2 score ({r2_score:.2f}) suggests that the regression model may need some adjustments. Consider the following:\n\n"
            suggestion += "- Adding additional relevant features to the model.\n"
            suggestion += "- Fine-tuning hyperparameters such as regularization strength or learning rate.\n"
            suggestion += "- Checking for multicollinearity among features and addressing it if present.\n"
        elif r2_score >= 0.5:
            suggestion = f"The R2 score ({r2_score:.2f}) indicates a moderate fit for the regression model. To improve the performance, you may:\n\n"
            suggestion += "- Experiment with different algorithms or ensemble methods.\n"
            suggestion += "- Feature engineering to create new informative features.\n"
            suggestion += "- Addressing outliers or anomalies in the data.\n"
        else:
            suggestion = f"The R2 score ({r2_score:.2f}) indicates a poor fit for the regression model. Consider the following steps to enhance the model performance:\n\n"
            suggestion += "- Exploring more complex models or nonlinear transformations of features.\n"
            suggestion += "- Collecting additional relevant data to improve model generalization.\n"
            suggestion += "- Conducting thorough feature selection to retain only the most informative features.\n"

    return suggestion

# Function to provide suggestions for classification model based on accuracy
def threshold_classification(accuracy, threshold=0.7):
    if accuracy >= threshold:
        suggestion = f"The accuracy score ({accuracy:.2f}) indicates a good performance for the classification model. No further action is required."
    else:
        if accuracy >= 0.6:
            suggestion = f"The accuracy score ({accuracy:.2f}) suggests that the classification model may need some adjustments. Consider the following:\n\n"
            suggestion += "- Adding more data or augmenting existing data to improve model generalization.\n"
            suggestion += "- Experimenting with different algorithms or hyperparameters.\n"
            suggestion += "- Performing feature engineering to create more informative features.\n"
        elif accuracy >= 0.5:
            suggestion = f"The accuracy score ({accuracy:.2f}) indicates a moderate performance for the classification model. To improve the accuracy, you may:\n\n"
            suggestion += "- Fine-tune the model's hyperparameters using techniques such as grid search or random search.\n"
            suggestion += "- Conducting thorough feature selection to retain only the most relevant features.\n"
            suggestion += "- Handling class imbalance issues using techniques such as oversampling or undersampling.\n"
        else:
            suggestion = f"The accuracy score ({accuracy:.2f}) suggests a poor performance for the classification model. Consider the following steps to enhance the model performance:\n\n"
            suggestion += "- Collecting more labeled data to improve model training.\n"
            suggestion += "- Exploring different machine learning algorithms or ensemble methods.\n"
            suggestion += "- Addressing data preprocessing issues such as feature scaling or normalization.\n"

    return suggestion

# Function to provide suggestions for clustering based on silhouette scores
def threshold_clustering(silhouette_scores, threshold=0.5):
    suggestions = []
    for silhouette_score in silhouette_scores:
        if silhouette_score >= threshold:
            suggestion = f"The silhouette score ({silhouette_score:.2f}) indicates good cluster separation."
        else:
            suggestion = f"The silhouette score ({silhouette_score:.2f}) suggests poor cluster separation. Consider the following:\n\n"
            suggestion += "- Experimenting with different clustering algorithms.\n"
            suggestion += "- Tuning the hyperparameters of the chosen algorithm.\n"
            suggestion += "- Evaluating feature selection and engineering methods.\n"
        suggestions.append(suggestion)
    return suggestions

# Main function for Streamlit app
def main(state, comp_table=None, X=None, y=None, df=None,model_names=None,models = None):
    if state == 1:  # Regression task
        st.subheader("Bar Graph")
        st.bar_chart(comp_table)
        st.subheader("Line Graph")
        st.line_chart(comp_table)
        st.subheader("Area Graph")
        st.area_chart(comp_table)
        st.subheader("Heatmap")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        sns.heatmap(comp_table, annot=True)
        st.pyplot()
        
        st.sidebar.subheader("Pick the Model You Want Suggestions")
        selected_model = st.sidebar.selectbox("Select Model", ["Best Model"] + model_names)

        if selected_model != "Best Model":
            r2_score = comp_table.loc[selected_model]["R2 score"]
            st.subheader(f"{selected_model} Model")
            st.write("R2 Score:", r2_score)
            suggestion = threshold_regression(r2_score)
            st.subheader("Model Performance Analysis")
            st.write(suggestion)
        else:
            max_index = comp_table["R2 score"].idxmax()
            r2_score = comp_table.loc[max_index]["R2 score"]
            st.subheader(f"{max_index} Model")
            st.write("This is the Best Model based on R2 Score")
            st.write("R2 Score:", r2_score)
    elif state == 2:  # Classification task
        # Classification visualizations
        st.header("Classification Visualizations")

        # Metrics comparison
        st.subheader("Metrics Comparison")

        colors = px.colors.qualitative.Plotly

        """ fig = px.bar(comp_table, x=comp_table.index, y=["Accuracy", "Precision", "Recall", "F1-score"], barmode="group", color_discrete_sequence=colors)
        fig.update_layout(title="Classification Metrics", xaxis_title="Model Name", yaxis_title="Metric Value")
        st.plotly_chart(fig)

        # Precision-Recall curve
        st.subheader("Precision-Recall Curve")
        fig = px.area(comp_table, x="Recall", y="Precision", color=comp_table.index, line_group=comp_table.index, color_discrete_sequence=colors)
        fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig) """

        array_cols = comp_table.dtypes[comp_table.dtypes == 'object'].index

        # Filter out columns with NumPy arrays
        for col in array_cols:
            if comp_table[col].apply(lambda x: isinstance(x, np.ndarray)).any():
                comp_table = comp_table.drop(col, axis=1)

        st.subheader("Bar Graph")
        st.bar_chart(comp_table)
        st.subheader("Line Graph")
        st.line_chart(comp_table)
        st.subheader("Area Graph")
        st.area_chart(comp_table)
        st.subheader("Heatmap")
        st.set_option('deprecation.showPyplotGlobalUse', False)

        non_numeric_cols = comp_table.select_dtypes(include=['object']).columns
        for col in non_numeric_cols:
            comp_table[col] = pd.to_numeric(comp_table[col], errors='coerce')

        # Replace any remaining non-numeric values with NaN
        comp_table = comp_table.replace([np.inf, -np.inf], np.nan)

        sns.heatmap(comp_table, annot=True)
        st.pyplot()

        st.sidebar.subheader("Pick the Model You Want Suggestions")
        selected_model = st.sidebar.selectbox("Select Model", ["Best Model"] + model_names)

        if selected_model != "Best Model":
            accuracy = comp_table.loc[selected_model]["Accuracy"]
            st.subheader(f"{selected_model} Model")
            st.write("Accuracy:", accuracy)
            suggestion = threshold_classification(accuracy)
            st.subheader("Model Performance Analysis")
            st.write(suggestion)
        else:
            comp_table["Accuracy"] = pd.to_numeric(comp_table["Accuracy"], errors='coerce')

            # Replace any remaining non-numeric values with NaN
            comp_table["Accuracy"] = comp_table["Accuracy"].replace([np.inf, -np.inf], np.nan)

            max_index = comp_table["Accuracy"].idxmax()
            accuracy = comp_table.loc[max_index]["Accuracy"]
            st.subheader(f"{max_index} Model")
            st.write("This is the Best Model based on Accuracy")
            st.write("Accuracy:", accuracy)

    elif state == 3:  # Clustering task
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.subheader("Elbow Method for Optimal Number of Clusters")
        nan_values = df.isna().sum().sum()

        if nan_values > 0:
            st.write("Warning: NaN values found in the data. Imputing NaN values with column means before proceeding.")
            df.fillna(df.mean(), inplace=True)

            
        distortions = []
        max_clusters = 10
        cluster_range = range(1, max_clusters + 1)

        for num_clusters in cluster_range:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            kmeans.fit(df)
            distortions.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(cluster_range, distortions, marker='o', linestyle='-')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Distortion')
        plt.title('Elbow Method for KMeans Clustering')
        plt.xticks(cluster_range)
        st.pyplot()

        # Calculate silhouette score
        st.subheader("Silhouette Score for KMeans Clustering")
        silhouette_scores = []
        for num_clusters in cluster_range:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(df)
            if len(np.unique(cluster_labels)) < 2:
                silhouette_scores.append(np.nan)  # Skip silhouette score calculation
            else:
                silhouette_avg = silhouette_score(df, cluster_labels)
                silhouette_scores.append(silhouette_avg)

        plt.figure(figsize=(10, 6))
        plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='-')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score for KMeans Clustering')
        plt.xticks(cluster_range)
        st.pyplot()

        # Threshold analysis
        st.subheader("Threshold Analysis for KMeans Clustering")
        suggestions = threshold_clustering(silhouette_scores)
        for k, suggestion in enumerate(suggestions, start=2):
            st.write(suggestion)

        st.write("Agglomerative Clustering is hierarchical and does not typically use the elbow method.")
        st.write("You may visualize the resulting dendrogram instead.")
        
        agglomerative = AgglomerativeClustering().fit(df)
        plt.figure(figsize=(10, 6))
        dendrogram = shc.dendrogram(shc.linkage(df, method='ward'))
        plt.title('Dendrogram for Agglomerative Clustering')
        st.pyplot()

        # Calculate silhouette score
        st.subheader("Silhouette Score for Agglomerative Clustering")
        silhouette_scores = []
        max_clusters = 10
        cluster_range = range(2, max_clusters + 1)
        for num_clusters in cluster_range:
            agg = AgglomerativeClustering(n_clusters=num_clusters)
            cluster_labels = agg.fit_predict(df)
            if len(np.unique(cluster_labels)) < 2:
                silhouette_scores.append(np.nan)  # Skip silhouette score calculation
            else:
                silhouette_avg = silhouette_score(df, cluster_labels)
                silhouette_scores.append(silhouette_avg)

        plt.figure(figsize=(10, 6))
        plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='-')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score for Agglomerative Clustering')
        plt.xticks(cluster_range)
        st.pyplot()

        # Threshold analysis
        st.subheader("Threshold Analysis for Agglomerative Clustering")
        suggestions = threshold_clustering(silhouette_scores)
        for k, suggestion in enumerate(suggestions, start=2):
            st.write(suggestion)


        st.write("DBSCAN does not require specifying the number of clusters.")
        st.write("Hence, Elbow Method is not applicable.")

        # Calculate silhouette score for DBSCAN
        st.subheader("Silhouette Score for DBSCAN Clustering")
        silhouette_scores = []
        # Take eps_values and min_samples_values from data
        eps_values = np.linspace(0.1, 2.0, num=10)
        min_samples_values = range(2, 10)
        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                cluster_labels = dbscan.fit_predict(df)
                if len(np.unique(cluster_labels)) < 2:
                    silhouette_scores.append(np.nan)  # Skip silhouette score calculation
                else:
                    silhouette_avg = silhouette_score(df, cluster_labels)
                    silhouette_scores.append(silhouette_avg)

        # Reshape silhouette scores for plotting
        silhouette_scores = np.array(silhouette_scores).reshape(len(eps_values), len(min_samples_values))

        plt.figure(figsize=(10, 6))
        for i, eps in enumerate(eps_values):
            plt.plot(min_samples_values, silhouette_scores[i], marker='o', label=f"Eps = {eps}")
        plt.xlabel('Min Samples')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score for DBSCAN Clustering')
        plt.legend()
        st.pyplot()

# # Initialize Streamlit app
# st.title("Data Visualization App")

# # Define state variable to switch between regression, classification, and clustering tasks
# state = st.sidebar.radio("Select Task", ["Regression", "Classification", "Clustering"])

# # Based on the selected task, execute the main function with appropriate parameters
# if state == "Regression":
#     main(1)
# elif state == "Classification":
#     main(2)
# elif state == "Clustering":
#     main(3)