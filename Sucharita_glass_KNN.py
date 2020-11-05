# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 00:06:26 2020

@author: SUCHARITA
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.metrics import classification_report, confusion_matrix
glass = pd.read_csv("F:\\ExcelR\\Assignment\\KNN\\glass.csv")
glass.shape # (214, 10)
glass.info()
glass['Type'].unique() # 7 types
glass.head()

# segregate dependent and independent variables
y= glass['Type'].values
x= glass.drop(['Type'],axis=1).values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 0)


# building models

knn= KNC(n_neighbors= 8)
# fit train data

knn.fit(x_train,y_train) # model built
y_train_pred = knn.predict(x_train) # predict "y_train" value based on model created
train_acc= np.mean(y_train_pred == y_train) # check accuracy of predicted and real value
train_acc # 73.68%
knn.score(x_train,y_train) # 73.68%

# check prediction accuracy of train data and classification error
print(confusion_matrix(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred))  # accuracy = 74%

# fit test data
test_acc = np.mean(knn.predict(x_test) == y_test)
test_acc # 58.13%
knn.score(x_test, y_test) # 58.13%
y_test_pred = knn.predict(x_test)

# check prediction accuracy of test data and classification error
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred)) # accuracy =58%
# as test accuracy much lower than train accuracy so we use new value of "k"
 
# for 5 neighbours
knn= KNC(n_neighbors= 5)
# fit train data
knn.fit(x_train,y_train)
y_train_pred = knn.predict(x_train)
train_acc= np.mean(knn.predict(x_train) == y_train)
train_acc # 77.19%
knn.score(x_train,y_train) # 77.19%
# check prediction accuracy of train data and classification error
print(confusion_matrix(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred))  # accuracy = 77%

# fit test data
test_acc = np.mean(knn.predict(x_test) == y_test)
test_acc # 58.13%
knn.score(x_test, y_test) # 58.13%
y_test_pred = knn.predict(x_test)

# check prediction accuracy of test data and classification error
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred)) # accuracy =58%

# as test accuracy much lower than train accuracy so we use new value of "k"

# for 3 neighbours
knn= KNC(n_neighbors= 3)
# fit train data
knn.fit(x_train,y_train)
y_train_pred = knn.predict(x_train)
train_acc= np.mean(knn.predict(x_train) == y_train)
train_acc # 83.62%
knn.score(x_train,y_train) # 83.62%

# check prediction accuracy of train data and classification error
print(confusion_matrix(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred)) # accuracy =84%

# fit test data
test_acc = np.mean(knn.predict(x_test) == y_test)
test_acc # 55.81%
knn.score(x_test, y_test) # 55.81%
y_test_pred = knn.predict(x_test)

# check prediction accuracy of test data and classification error
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred)) # accuracy =56%

# getting optimum "k" value
a= []
for i in range (1,10,2):
    knn = KNC(n_neighbors=i)
    knn.fit(x_train, y_train)
    train_acc = np.mean(knn.predict(x_train)==y_train)
    test_acc = np.mean(knn.predict(x_test)==y_test)
    a.append([train_acc,test_acc])

plt.plot(np.arange(1,10,2),[i[0] for i in a],"bo-")
plt.plot(np.arange(1,10,2), [i[1] for i in a], "rs-")
plt.legend(["train","test"])

# consider neighbour = 5 to be optimum
# model is already built above












