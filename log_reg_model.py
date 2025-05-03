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
