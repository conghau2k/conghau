import pandas as pd
from sklearn import datasets,tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv("iris.csv")
X = data[["sepal.length", "sepal.width", "petal.length", "petal.width"]]
y = data["variety"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=50,test_size=0.3)
clf = DecisionTreeClassifier(criterion='gini')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
pickle.dump(clf,open("model.pkl","wb"))
