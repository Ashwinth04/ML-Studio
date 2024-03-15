from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import svm
from sklearn.linear_model import LogisticRegression, LinearRegression
import pandas as pd


class Models:
    def __init__(self, dataframe):
        self.df = dataframe
        self.X = None
        self.y = None

    def clean_and_scale_dataset(self):
        self.df = self.df.dropna()
        self.X = self.df.drop(columns=self.df.columns[-1])
        self.y = self.df.iloc[:, -1]
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        return self.X, self.y

    def SVM(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        classifier = svm.SVC()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return classifier, accuracy

    def logistic_regression(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return classifier, accuracy

    def decision_tree(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        classifier = tree.DecisionTreeClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return classifier, accuracy

    def knn(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        classifier = KNeighborsClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return classifier, accuracy

    def linear_regression(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test, y_pred)
        return regressor, score

    def dtree_regressor(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        regressor = tree.DecisionTreeRegressor()
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test, y_pred)
        return regressor, score

    def SVR(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        regressor = svm.SVR()
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test, y_pred)
        return regressor, score

    def predict(self, clf, x):
        answer = clf.predict(x)
        return answer

    def pick_best_classifier(self):
        model_name, model, opt_score = "", None, 0

        classifiers = {
            "SVM": self.SVM(),
            "Logistic Regression": self.logistic_regression(),
            "Decision Tree": self.decision_tree(),
            "KNN": self.knn()
        }

        for name, tup in classifiers.items():
            clf, score = tup
            if (score > opt_score):
                opt_score = score
                model = clf
                model_name = name

        return model_name, model, opt_score