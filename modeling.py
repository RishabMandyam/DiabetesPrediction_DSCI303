# Log Reg

## Basic Logistic Regression SciKit-Learn Model

import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.metrics import accuracy_score
from IPython.display import display, Markdown
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv('diabetes_012_health_indicators.csv')

X = df.drop(columns='Diabetes_012')
y = df['Diabetes_012']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

model.predict(X_test)
model.score(X_test, y_test)

# Predict and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display fancy output
display(Markdown(f"""
#**Accuracy Score:**
## <span> {accuracy:.2%} </span>
"""))

precision = precision_score(y_test, y_pred, average=None)
print("\nPrecision:", precision[0])

recall = recall_score(y_test, y_pred, average=None)
print("Recall:", recall[0])

f1 = f1_score(y_test, y_pred, average=None)
print("F1-score:", f1[0])

print("Mean absolute error: ", mean_absolute_error(y_test, y_pred))
print("Mean error rate: ", average_error_rate(y_test, y_pred))
print()

# Binarize the output labels (assuming y_test contains original class labels)
y_test_bin = label_binarize(y_test, classes=np.unique(y))
n_classes = y_test_bin.shape[1]

# Predict probabilities
y_score = model.predict_proba(X_test)

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multiclass')
plt.legend(loc="lower right")
plt.show()



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

df = pd.read_csv('diabetes_012_health_indicators.csv')

X = df.drop(columns='Diabetes_012')
y = df.Diabetes_012

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 5)

sc = StandardScaler()
sc.fit(X_train)

X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# Figure out how how many neighbors to use (k)
k_list = list(range(1,50))

cv_scores = []

for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1-x for x in cv_scores]

plt.figure()
plt.figure(figsize=(15,10))
plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
plt.xlabel('Number of Neighbors K', fontsize=15)
plt.ylabel('Misclassification Error', fontsize=15)
sns.set_style("whitegrid")
plt.plot(k_list, MSE)
plt.show()

best_k = k_list[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d." % best_k)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,p=2,)
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



## SVM Model

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Create a 3-class classification problem
X = df.drop(columns='Diabetes_012')
y = df.Diabetes_012
X = StandardScaler().fit_transform(X)

# Fit the SVM model
model = SVC(kernel='linear', decision_function_shape='ovo')  # 'ovo' is default for multiclass
model.fit(X, y)

# Visualization
def plot_multiclass_decision_function(model, X, y, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca()
    xlim = (X[:, 0].min() - 1, X[:, 0].max() + 1)
    ylim = (X[:, 1].min() - 1, X[:, 1].max() + 1)

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 200),
                         np.linspace(ylim[0], ylim[1], 200))
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    Z = model.predict(xy).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='winter')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter', edgecolors='k')

    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=100, linewidth=1, facecolors='none', edgecolors='red')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title("Multiclass SVM (linear kernel)")
    return scatter

plt.figure(figsize=(8, 6))
plot_multiclass_decision_function(model, X, y)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
