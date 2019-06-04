import numpy as np
import pandas
from sklearn.tree import DecisionTreeClassifier
'''
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])
clf = DecisionTreeClassifier()
clf.fit(X, y)
PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
'''


data = pandas.read_csv('titanic.csv')
dataMain = data.drop(data.columns[[0,3,6,7,8,10,11]], axis = 1)
dataMain = dataMain.dropna()
survived = dataMain['Survived']
signs = dataMain.drop(dataMain.columns[[0]], axis = 1)
signs = signs.replace('male', 1)
signs = signs.replace('female', 0)
signs = signs.dropna()
clf = DecisionTreeClassifier(random_state=241)
clf.fit(signs,survived)
importances = clf.feature_importances_
print(importances)
