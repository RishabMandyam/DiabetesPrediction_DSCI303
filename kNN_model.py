## kNN Model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier # Import KNeighborsClassifier
import seaborn as sns

df = pd.read_csv('diabetes_binary_5050split_health_indicators.csv')

X = df.drop(columns='Diabetes_binary')
y = df.Diabetes_binary

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 5)

sc = StandardScaler()
sc.fit(X_train)

X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# Figure out how how many neighbors to use (k)
k_list = list(range(40,60))

cv_scores = []

for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1-x for x in cv_scores]

best_k = k_list[MSE.index(min(MSE))]

plt.figure()
plt.figure(figsize=(15,10))
plt.title('The optimal number of neighbors: %d' % best_k, fontsize=20, fontweight='bold')
plt.xlabel('Number of Neighbors K', fontsize=15)
plt.ylabel('Misclassification Error', fontsize=15)
sns.set_style("whitegrid")
plt.plot(k_list, MSE)
plt.show()

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=55,p=2,)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
print(y_pred.shape,y_test.shape)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of the model is ' + str(round(accuracy, 2)) + ' %.')

# classification performance for test data
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
