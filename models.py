from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error,median_absolute_error, explained_variance_score, precision_score, recall_score,f1_score,silhouette_score,davies_bouldin_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import tree
from sklearn import svm
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import os

class Models:
    def __init__(self,dataframe):
        self.df = dataframe
        self.X = None
        self.y = None

    def clean_and_scale_dataset(self):
        self.df = self.df.dropna()
        self.X = self.df.drop(columns = self.df.columns[-1])
        self.y = self.df.iloc[:,-1]
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        return self.X,self.y
    
    ####################### Classifiers ################################
    def SVM(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        classifier = svm.SVC()
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred,average = "micro")
        recall = recall_score(y_test,y_pred,average = "micro")
        f1score = f1_score(y_test,y_pred,average = "micro")
        cross_val_scores = cross_val_score(classifier,self.X,self.y,cv = k)
        metrics = {
            "Accuracy":accuracy,
            "Precision":precision,
            "Recall":recall,
            "F1-score":f1score,
            "Cross val scores":cross_val_scores
        }
        return classifier,metrics
    
    def logistic_regression(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        classifier =  LogisticRegression()
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred,average = "micro")
        recall = recall_score(y_test,y_pred,average = "micro")
        f1score = f1_score(y_test,y_pred,average = "micro")
        cross_val_scores = cross_val_score(classifier,self.X,self.y,cv = k)
        metrics = {
            "Accuracy":accuracy,
            "Precision":precision,
            "Recall":recall,
            "F1-score":f1score,
            "Cross val scores":cross_val_scores
        }
        return classifier,metrics
    
    def dtree_classifier(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        classifier = tree.DecisionTreeClassifier()
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred,average = "micro")
        recall = recall_score(y_test,y_pred,average = "micro")
        f1score = f1_score(y_test,y_pred,average = "micro")
        cross_val_scores = cross_val_score(classifier,self.X,self.y,cv = k)
        metrics = {
            "Accuracy":accuracy,
            "Precision":precision,
            "Recall":recall,
            "F1-score":f1score,
            "Cross val scores":cross_val_scores
        }
        return classifier,metrics
    
    def knn_classifier(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        classifier = KNeighborsClassifier()
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred,average = "micro")
        recall = recall_score(y_test,y_pred,average = "micro")
        f1score = f1_score(y_test,y_pred,average = "micro")
        cross_val_scores = cross_val_score(classifier,self.X,self.y,cv = k)
        metrics = {
            "Accuracy":accuracy,
            "Precision":precision,
            "Recall":recall,
            "F1-score":f1score,
            "Cross val scores":cross_val_scores
        }
        return classifier,metrics
    
    def naivebayes(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        classifier = GaussianNB()
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_pred,y_test)
        precision = precision_score(y_test,y_pred,average = "micro")
        recall = recall_score(y_test,y_pred,average = "micro")
        f1score = f1_score(y_test,y_pred,average = "micro")
        cross_val_scores = cross_val_score(classifier,self.X,self.y,cv = k)
        metrics = {
            "Accuracy":accuracy,
            "Precision":precision,
            "Recall":recall,
            "F1-score":f1score,
            "Cross val scores":cross_val_scores
        }
        return classifier,metrics
    
    def adaboost(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        classifier = AdaBoostClassifier()
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_pred,y_test)
        precision = precision_score(y_test,y_pred,average = "micro")
        recall = recall_score(y_test,y_pred,average = "micro")
        f1score = f1_score(y_test,y_pred,average = "micro")
        cross_val_scores = cross_val_score(classifier,self.X,self.y,cv = k)
        metrics = {
            "Accuracy":accuracy,
            "Precision":precision,
            "Recall":recall,
            "F1-score":f1score,
            "Cross val scores":cross_val_scores
        }
        return classifier,metrics
    
    def mlp(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        classifier = MLPClassifier()
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_pred,y_test)
        precision = precision_score(y_test,y_pred,average = "micro")
        recall = recall_score(y_test,y_pred,average = "micro")
        f1score = f1_score(y_test,y_pred,average = "micro")
        cross_val_scores = cross_val_score(classifier,self.X,self.y,cv = k)
        metrics = {
            "Accuracy":accuracy,
            "Precision":precision,
            "Recall":recall,
            "F1-score":f1score,
            "Cross val scores":cross_val_scores
        }
        return classifier,metrics
    
    def gradient_boost(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        classifier = GradientBoostingClassifier()
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_pred,y_test)
        precision = precision_score(y_test,y_pred,average = "micro")
        recall = recall_score(y_test,y_pred,average = "micro")
        f1score = f1_score(y_test,y_pred,average = "micro")
        cross_val_scores = cross_val_score(classifier,self.X,self.y,cv = k)
        metrics = {
            "Accuracy":accuracy,
            "Precision":precision,
            "Recall":recall,
            "F1-score":f1score,
            "Cross val scores":cross_val_scores
        }
        return classifier,metrics
    
    def random_forest_classifier(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        classifier = RandomForestClassifier()
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_pred,y_test)
        precision = precision_score(y_test,y_pred,average = "micro")
        recall = recall_score(y_test,y_pred,average = "micro")
        f1score = f1_score(y_test,y_pred,average = "micro")
        cross_val_scores = cross_val_score(classifier,self.X,self.y,cv = k)
        metrics = {
            "Accuracy":accuracy,
            "Precision":precision,
            "Recall":recall,
            "F1-score":f1score,
            "Cross val scores":cross_val_scores
        }
        return classifier,metrics

    ##########################  Regressors ##################################
    def linear_regression(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        regressor = LinearRegression()
        regressor.fit(X_train,y_train)
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        mde = median_absolute_error(y_test,y_pred)
        evs = explained_variance_score(y_test,y_pred,force_finite=False)
        cross_val_scores = cross_val_score(regressor,self.X,self.y,cv = k)
        metrics = {
            "R2 score": score,
            "Mean Absolute Error":mae,
            "Median Absolute Error":mde,
            "Explained Variance Score":evs,
            "Cross val scores":cross_val_scores
        }
        return regressor,metrics
    
    def dtree_regressor(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        regressor = tree.DecisionTreeRegressor()
        regressor.fit(X_train,y_train)
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        mde = median_absolute_error(y_test,y_pred)
        evs = explained_variance_score(y_test,y_pred,force_finite=False)
        cross_val_scores = cross_val_score(regressor,self.X,self.y,cv = k)
        metrics = {
            "R2 score": score,
            "Mean Absolute Error":mae,
            "Median Absolute Error":mde,
            "Explained Variance Score":evs,
            "Cross val scores":cross_val_scores
        }
        return regressor,metrics
    
    def SVR(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        regressor = svm.SVR()
        regressor.fit(X_train,y_train)
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        mde = median_absolute_error(y_test,y_pred)
        evs = explained_variance_score(y_test,y_pred,force_finite=False)
        cross_val_scores = cross_val_score(regressor,self.X,self.y,cv = k)
        metrics = {
            "R2 score": score,
            "Mean Absolute Error":mae,
            "Median Absolute Error":mde,
            "Explained Variance Score":evs,
            "Cross val scores":cross_val_scores
        }
        return regressor,metrics
    
    def ridge_regression(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        regressor = Ridge()
        regressor.fit(X_train,y_train)
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        mde = median_absolute_error(y_test,y_pred)
        evs = explained_variance_score(y_test,y_pred,force_finite=False)
        cross_val_scores = cross_val_score(regressor,self.X,self.y,cv = k)
        metrics = {
            "R2 score": score,
            "Mean Absolute Error":mae,
            "Median Absolute Error":mde,
            "Explained Variance Score":evs,
            "Cross val scores":cross_val_scores
        }
        return regressor,metrics
    
    def lasso(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        regressor = Lasso()
        regressor.fit(X_train,y_train)
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        mde = median_absolute_error(y_test,y_pred)
        evs = explained_variance_score(y_test,y_pred,force_finite=False)
        cross_val_scores = cross_val_score(regressor,self.X,self.y,cv = k)
        metrics = {
            "R2 score": score,
            "Mean Absolute Error":mae,
            "Median Absolute Error":mde,
            "Explained Variance Score":evs,
            "Cross val scores":cross_val_scores
        }
        return regressor,metrics
    
    def elasticnet(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        regressor = ElasticNet()
        regressor.fit(X_train,y_train)
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        mde = median_absolute_error(y_test,y_pred)
        evs = explained_variance_score(y_test,y_pred,force_finite=False)
        cross_val_scores = cross_val_score(regressor,self.X,self.y,cv = k)
        metrics = {
            "R2 score": score,
            "Mean Absolute Error":mae,
            "Median Absolute Error":mde,
            "Explained Variance Score":evs,
            "Cross val scores":cross_val_scores
        }
        return regressor,metrics
    
    def random_forest_regressor(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        regressor = RandomForestRegressor()
        regressor.fit(X_train,y_train)
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        mde = median_absolute_error(y_test,y_pred)
        evs = explained_variance_score(y_test,y_pred,force_finite=False)
        cross_val_scores = cross_val_score(regressor,self.X,self.y,cv = k)
        metrics = {
            "R2 score": score,
            "Mean Absolute Error":mae,
            "Median Absolute Error":mde,
            "Explained Variance Score":evs,
            "Cross val scores":cross_val_scores
        }
        return regressor,metrics
    
    def mlp_regressor(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        regressor = MLPRegressor()
        regressor.fit(X_train,y_train)
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        mde = median_absolute_error(y_test,y_pred)
        evs = explained_variance_score(y_test,y_pred,force_finite=False)
        cross_val_scores = cross_val_score(regressor,self.X,self.y,cv = k)
        metrics = {
            "R2 score": score,
            "Mean Absolute Error":mae,
            "Median Absolute Error":mde,
            "Explained Variance Score":evs,
            "Cross val scores":cross_val_scores
        }
        return regressor,metrics
    
    def knn_regressor(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        regressor = KNeighborsRegressor()
        regressor.fit(X_train,y_train)
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        mde = median_absolute_error(y_test,y_pred)
        evs = explained_variance_score(y_test,y_pred,force_finite=False)
        cross_val_scores = cross_val_score(regressor,self.X,self.y,cv = k)
        metrics = {
            "R2 score": score,
            "Mean Absolute Error":mae,
            "Median Absolute Error":mde,
            "Explained Variance Score":evs,
            "Cross val scores":cross_val_scores
        }
        return regressor,metrics
    
    def gradient_boost_regressor(self,k = 5):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        regressor = GradientBoostingRegressor()
        regressor.fit(X_train,y_train)
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        mde = median_absolute_error(y_test,y_pred)
        evs = explained_variance_score(y_test,y_pred,force_finite=False)
        cross_val_scores = cross_val_score(regressor,self.X,self.y,cv = k)
        metrics = {
            "R2 score": score,
            "Mean Absolute Error":mae,
            "Median Absolute Error":mde,
            "Explained Variance Score":evs,
            "Cross val scores":cross_val_scores
        }
        return regressor,metrics
    
    ########################## Clustering ####################################

    def kmeans(self,num_clusters = 5):
        X_scaled, _ = self.clean_and_scale_dataset()
        sil_scores = []

        if num_clusters == -1:
            for i in range(2, 11):
                kmeans = KMeans(n_clusters=i, random_state=42)
                labels = kmeans.fit_predict(X_scaled)
                sil_score = silhouette_score(X_scaled, labels)
                sil_scores.append(sil_score)
            num_clusters = np.argmax(sil_scores) + 2
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
        else:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled,labels)
        db_score = davies_bouldin_score(X_scaled,labels)
        metrics = {
            "Silhouette score":sil_score,
            "Davies Bouldin score":db_score
        }
        return kmeans, labels, metrics, num_clusters
    
    def agglomerative_clustering(self,num_clusters = 5):
        X_scaled, _ = self.clean_and_scale_dataset()
        sil_scores = []

        if num_clusters == -1:
            for i in range(2, 11):
                agg = AgglomerativeClustering(n_clusters=i)
                labels = agg.fit_predict(X_scaled)
                sil_score = silhouette_score(X_scaled, labels)
                sil_scores.append(sil_score)
            num_clusters = np.argmax(sil_scores) + 2
            agg = AgglomerativeClustering(n_clusters=num_clusters)
            labels = agg.fit_predict(X_scaled)
        else:
            agg = AgglomerativeClustering(n_clusters=num_clusters)
            labels = agg.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled,labels)
        db_score = davies_bouldin_score(X_scaled,labels)
        metrics = {
            "Silhouette score":sil_score,
            "Davies Bouldin score":db_score
        }
        return agg, labels, metrics, num_clusters
    
    def dbscan(self,eps=2, min_samples = 5):
        X_scaled, _ = self.clean_and_scale_dataset()
        dbscan = DBSCAN(eps = eps,min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled,labels)
        db_score = davies_bouldin_score(X_scaled,labels)
        metrics = {
            "Silhouette score":sil_score,
            "Davies Bouldin score":db_score
        }
        return dbscan, labels, metrics

    # def predict(self,model,x):
    #     answer = model.predict(x)
    #     return answer
    
    def pick_best_classifier(self):
        model_name,model,opt_score = "",None,0.0

        classifiers = {
            "SVM":self.SVM(),
            "Logistic Regression":self.logistic_regression(),
            "Decision Tree": self.dtree_classifier(),
            "KNN":self.knn_classifier(),
            "Naive Bayes": self.naivebayes(),
            "Ada boost": self.adaboost(),
            "Gradient boost": self.gradient_boost(),
            "Random Forest": self.random_forest_classifier(),
        }      
        
        for name,tup in classifiers.items():
            clf,score = tup
            # print(score['Accuracy'])
            if(score['Accuracy'] > opt_score):
                opt_score = score['Accuracy']
                model = clf
                model_name = name

        return classifiers, model_name, model, opt_score
    
    def pick_best_regressor(self):
        model_name, model, opt_score = "",None,0

        regressors = {
            "Linear Regression": self.linear_regression(),
            "Decision tree regression": self.dtree_regressor(),
            "Support vector regression": self.SVR(),
            "Ridge regression":self.ridge_regression(),
            "Lasso regression":self.lasso(),
            "ElasticNet regression":self.elasticnet(),
            "Random Forest regression":self.random_forest_regressor(),
            "Multi-layer perceptron":self.mlp_regressor(),
            "KNN":self.knn_regressor(),
            "Gradient Boosting regression":self.gradient_boost_regressor()
        }

        for name,tup in regressors.items():
            clf,score = tup
            if(score['R2 score'] > opt_score):
                opt_score = score['R2 score']
                model = clf
                model_name = name

        return regressors, model_name, model, opt_score
    
    def return_all_clusters(self):
        pass

# df = pd.read_csv(os.path.join('iris.data.csv'))
# # print(df)
# s = Models(df)
# s.clean_and_scale_dataset()
# # classifiers,model_name,model,opt = s.pick_best_regressor()
# # print(classifiers)
# model,labels,metrics,k = s.kmeans(-1)
# print(metrics,k)