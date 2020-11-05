# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 01:29:52 2020

@author: SUCHARITA
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
import matplotlib.pyplot as plt
zoo = pd.read_csv("F:\\ExcelR\\Assignment\\KNN\\Zoo.csv")
zoo.shape
zoo.info()
zoo['type'].unique()
zoo.head()
y= zoo['type'].values
x= zoo.drop(['type','animal_name'],axis=1).values
x_train,x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

# knn.fit(x, y)
# for 7 neighbours

knn = KNC(n_neighbors=7)
# fit train data
knn.fit(x_train,y_train)
y_train_pred = knn.predict(x_train)
train_acc= np.mean(y_train_pred == y_train)
train_acc # 85.71%
knn.score(x_train,y_train) # 88.75%

# check prediction accuracy of train data and classification error
print(confusion_matrix(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred)) # accuracy = 89%

# fit test data
test_acc= np.mean(knn.predict(x_test)== y_test)
test_acc # 87.09%
knn.score(x_test,y_test) # 100%
y_test_pred = knn.predict(x_test)
# check prediction accuracy of test data and classification error
print(confusion_matrix(y_test, y_test_pred)) # no misclassification
print(classification_report(y_test, y_test_pred))  # accuracy = 100%

# for 4 neighbors
knn = KNC(n_neighbors= 4)
# fit train data
knn.fit(x_train, y_train)
y_train_pred= knn.predict(x_train)
train_acc = np.mean((y_train_pred) == (y_train))
train_acc # 97.5%
knn.score(x_train, y_train) # 97.5%

# check prediction accuracy of train data and classification error
print(confusion_matrix(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred)) # accuracy = 97%

# fit test data
test_acc= np.mean(knn.predict(x_test)==y_test)
test_acc # 100%
knn.score(x_test, y_test) # 100%
y_test_pred = knn.predict(x_test)
# check prediction accuracy of test data and classification error
print(confusion_matrix(y_test, y_test_pred)) # no misclassification
print(classification_report(y_test, y_test_pred))  # accuracy = 100%


# getting optimal "k" value
a= []
for i in range (3,50,2):
    knn = KNC(n_neighbors=i)
    knn.fit(x_train, y_train)
    train_acc = np.mean(knn.predict(x_train)==y_train)
    test_acc = np.mean(knn.predict(x_test)==y_test)
    a.append([train_acc,test_acc])

plt.plot(np.arange(3,50,2),[i[0] for i in a],"bo-")
plt.plot(np.arange(3,50,2), [i[1] for i in a], "rs-")
plt.legend(["train","test"])


# optimal value of "neighbours" = 4
# model is already built above