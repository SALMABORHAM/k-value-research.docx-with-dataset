# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 20:17:25 2020

@author: Reem Elsamahy
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


headernames = ['sl_no',	'ssc_p'	,	'hsc_p'		,	'degree_p	'	,	'etest_p'		,'mba_p'	,	'salary','gender']

dataset = pd.read_csv("C:/Users/Reem Elsamahy/Desktop/AAST Subjects/Professional Training/Dataset/Placement_Data_Full_Class.csv", names = headernames)
dataset.head()

X = dataset.iloc[:, 0:7].values
y = dataset.iloc[:, 7].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# how to find optimal value of k 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
# Creating odd list K for KNN
neighbors = list(range(1,50,2))
# empty list that will hold cv scores
cv_scores = [ ]

#perform 10-fold cross-validation
for K in neighbors:
    knn = KNeighborsClassifier(n_neighbors = K)
    scores = cross_val_score(knn,X_train,y_train,cv = 10,scoring ="accuracy")
    cv_scores.append(scores.mean())

# Changing to mis classification error
mse = [1-x for x in cv_scores]
# determing best k
optimal_k = neighbors[mse.index(min(mse))]
print("The optimal no. of neighbors is {}".format(optimal_k))

import matplotlib.pyplot as plt
def plot_accuracy(knn_list_scores):
    pd.DataFrame({"K":[i for i in range(1,50,2)], "Accuracy":knn_list_scores}).set_index("K").plot.bar(figsize= (9,6),ylim=(0.78,0.83),rot=0)
    plt.show()
#plot_accuracy(cv_scores)



from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 1)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)




