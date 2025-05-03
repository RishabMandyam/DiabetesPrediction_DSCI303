## Neural Network Model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import mglearn

# Load dataset
df = pd.read_csv('diabetes_012_health_indicators.csv')

# Select only two features for visualization
X = df[['BMI', 'Age']].values
y = df['Diabetes_012'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
mlp = MLPClassifier(solver='adam', random_state=0, max_iter=500).fit(X_train_scaled, y_train)

# Plot decision boundary (works only with 2D input)
mglearn.plots.plot_2d_separator(mlp, X_train_scaled, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], y_train)
plt.xlabel("BMI")
plt.ylabel("Age")
plt.title("Decision Boundary (2-feature MLP)")
plt.show()

# Accuracy
print("Training set score: %f" % mlp.score(X_train_scaled, y_train))
print("Test set score: %f" % mlp.score(X_test_scaled, y_test))

# Loss curve
plt.figure()
plt.plot(mlp.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.show()

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
import mglearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X = df[['BMI', 'Age']].values
y = df['Diabetes_012'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    random_state=42)

mlp = MLPClassifier(solver='adam', random_state=0, max_iter=10000).fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

#feed test data into models
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

# plot loss curve
# new figure
plt.figure()
plt.plot(mlp.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Loss")

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('diabetes_binary_5050split_health_indicators.csv')

# Assuming df is your DataFrame
X = df.drop(columns='Diabetes_binary')
y = df.Diabetes_binary

# Standardize the data - important for neural networks
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Training and test set
X_train_, X_test, y_train_, y_test = train_test_split(X, y, stratify=y,
                                                    random_state=42)

# Validation set
X_train, X_val, y_train, y_val = train_test_split(X_train_, y_train_, stratify=y_train_,
                                                    random_state=42)

# Sample a smaller subset for visualization only
# Using 5000 samples or fewer for visualization to save memory
max_viz_samples = min(5000, len(X_train))
viz_indices = np.random.choice(len(X_train), max_viz_samples, replace=False)
X_train_viz = X_train.iloc[viz_indices] if isinstance(X_train, pd.DataFrame) else X_train[viz_indices]
y_train_viz = y_train.iloc[viz_indices] if isinstance(y_train, pd.Series) else y_train[viz_indices]

# Apply PCA to reduce data to 2D for visualization purposes
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train_viz)

# For the final visualization
max_viz_samples_final = min(5000, len(X_train_))
viz_indices_final = np.random.choice(len(X_train_), max_viz_samples_final, replace=False)
X_train_viz_final = X_train_.iloc[viz_indices_final] if isinstance(X_train_, pd.DataFrame) else X_train_[viz_indices_final]
y_train_viz_final = y_train_.iloc[viz_indices_final] if isinstance(y_train_, pd.Series) else y_train_[viz_indices_final]
X_train_2d_final = pca.transform(X_train_viz_final)

# Grid search - Define parameters
learning_rates = [0.001, 0.01, 0.1]
hidden_layer_sizes = [(10,), (50,), (100,), (10, 10), (50, 50), (100, 100)]
best_score = 0
best_params = {}

# Configure plots - use fewer subplots at a time
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Memory-efficient approach for grid search
for idx, hls in enumerate(hidden_layer_sizes):
    # Create a new figure for each hidden layer size to free up memory
    if idx > 0 and idx % 3 == 0:
        plt.tight_layout()
        plt.savefig(f'mlp_grid_search_batch_{idx//3}.png')
        plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
    
    for j, lr in enumerate(learning_rates):
        ax_idx = j % 3 + (idx % 2) * 3  # Calculate current axis index
        ax = axes[ax_idx]
        
        print(f"Training MLP with hidden_layer_sizes={hls}, learning_rate={lr}")
        
        # Use early stopping for faster convergence and memory efficiency
        mlp = MLPClassifier(
            solver='adam', 
            random_state=0, 
            hidden_layer_sizes=hls, 
            learning_rate_init=lr, 
            max_iter=1000,
            early_stopping=True,  # Use validation set for early stopping
            validation_fraction=0.1,
            n_iter_no_change=10,  # Stop if no improvement for 10 iterations
            batch_size=min(200, len(X_train))  # Use mini-batches
        )
        
        mlp.fit(X_train, y_train)
        
        # Create a 2D mesh grid for visualization - use a coarser grid to save memory
        h = 0.1  # Increased step size for less memory usage
        x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
        y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
        
        # Create a coarser mesh grid
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h)
        )
        
        # Process mesh in smaller chunks to save memory
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        chunk_size = 1000  # Process 1000 points at a time
        Z = np.zeros(mesh_points.shape[0])
        
        for i in range(0, mesh_points.shape[0], chunk_size):
            end = min(i + chunk_size, mesh_points.shape[0])
            chunk = mesh_points[i:end]
            # Transform back to original feature space
            chunk_original = pca.inverse_transform(chunk)
            # Predict
            Z[i:end] = mlp.predict(chunk_original)
        
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary
        ax.contourf(xx, yy, Z, alpha=0.3)
        ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train_viz, edgecolors='k', s=20)
        
        if len(hls) > 1:
            ax.set_title(f"n_hidden=[{hls[0]}, {hls[1]}]\nlr={lr:.4f}")
        else:
            ax.set_title(f"n_hidden=[{hls[0]}]\nlr={lr:.4f}")

        # Check if this model is the best so far
        score = mlp.score(X_val, y_val)
        if score > best_score:
            best_score = score
            best_params = {'learning_rate': lr, 'hidden_layer_sizes': hls}
            print(f"Best score: {best_score}")
            print(f"Best params: {best_params}")
            print(f"Val set score: {score}")
            print(f"Training set score: {mlp.score(X_train, y_train)}")
        
        # Clear some memory
        del mlp
        import gc
        gc.collect()

# Save last batch of plots
plt.tight_layout()
plt.savefig(f'mlp_grid_search_batch_final.png')
plt.close(fig)

print(f"Best parameters found: {best_params}")
print(f"Best validation score: {best_score}")

# Train the final model with best parameters on full training set
print("Training final model with best parameters...")
mlp = MLPClassifier(
    solver='adam', 
    random_state=0, 
    hidden_layer_sizes=best_params['hidden_layer_sizes'], 
    learning_rate_init=best_params['learning_rate'], 
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    batch_size=min(200, len(X_train_))  # Use mini-batches
)
mlp.fit(X_train_, y_train_)

test_score = mlp.score(X_test, y_test)
print(f"Test set score: {test_score}")

# Plot the final model's decision boundary with a coarser grid
fig, ax = plt.subplots(figsize=(10, 8))
h = 0.1  # Coarser grid
x_min, x_max = X_train_2d_final[:, 0].min() - 1, X_train_2d_final[:, 0].max() + 1
y_min, y_max = X_train_2d_final[:, 1].min() - 1, X_train_2d_final[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

# Process mesh in chunks
mesh_points = np.c_[xx.ravel(), yy.ravel()]
chunk_size = 1000
Z = np.zeros(mesh_points.shape[0])

for i in range(0, mesh_points.shape[0], chunk_size):
    end = min(i + chunk_size, mesh_points.shape[0])
    chunk = mesh_points[i:end]
    chunk_original = pca.inverse_transform(chunk)
    Z[i:end] = mlp.predict(chunk_original)

Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, alpha=0.3)
ax.scatter(X_train_2d_final[:, 0], X_train_2d_final[:, 1], c=y_train_viz_final, edgecolors='k', s=20)

if len(best_params['hidden_layer_sizes']) > 1:
    ax.set_title(f"Final model: n_hidden=[{best_params['hidden_layer_sizes'][0]}, "
                f"{best_params['hidden_layer_sizes'][1]}]\n"
                f"lr={best_params['learning_rate']:.4f}, Test score: {test_score:.4f}")
else:
    ax.set_title(f"Final model: n_hidden=[{best_params['hidden_layer_sizes'][0]}]\n"
                f"lr={best_params['learning_rate']:.4f}, Test score: {test_score:.4f}")

plt.tight_layout()
plt.savefig('final_model.png')

# Plot loss curve if available
if hasattr(mlp, 'loss_curve_'):
    plt.figure(figsize=(8, 6))
    plt.plot(mlp.loss_curve_)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Curve for Best Model")
    plt.tight_layout()
    plt.savefig('loss_curve.png')

plt.close('all')  # Close all figures to free memory

print("All plots have been saved as PNG files")
print("Training and evaluation complete!")

# Optional: Save the trained model
import joblib
joblib.dump(mlp, 'diabetes_mlp_model.joblib')
print("Model saved to 'diabetes_mlp_model.joblib'")

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('diabetes_012_health_indicators.csv')

# Extract features and target
X = df.drop(columns='Diabetes_012')
y = df.Diabetes_012

# Standardize the features - important for neural networks
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=1, stratify=y)

# Create and train the classifier
clf = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # Two hidden layers with 100 and 50 neurons
    activation='relu',
    solver='adam',
    alpha=0.0001,  # L2 regularization parameter
    batch_size='auto',
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=1
).fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
print(f"Predictions for first 5 samples: {y_pred[:5]}")

# Evaluate the model
print("Training set score: %f" % clf.score(X_train, y_train))
print("Test set score: %f" % clf.score(X_test, y_test))

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('MLP Classifier: Actual vs Predicted')
plt.grid(True)
plt.show()

# Optional: Visualize confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
classes = np.unique(y)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Add text annotations to each cell
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()
