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
