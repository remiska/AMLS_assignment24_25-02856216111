
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV,  learning_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the Datasets folder relative to the script location
data_path = os.path.join(script_dir, "..", "Datasets", "bloodmnist.npz")

# Load BloodMNIST dataset from the local .npz file
data = np.load(data_path)


# Extract images and labels
X_train = data["train_images"]  # Training images
y_train = data["train_labels"]  # Training labels
X_val = data["val_images"]      # Validation images
y_val = data["val_labels"]      # Validation labels
X_test = data["test_images"]    # Test images
y_test = data["test_labels"]    # Test labels

# Combine training and validation sets for training
X_train = np.concatenate([X_train, X_val], axis=0)
y_train = np.concatenate([y_train, y_val], axis=0)

# Ensure y_train and y_test are 1D arrays
y_train = y_train.ravel()  # Flatten the labels to be 1D
y_test = y_test.ravel()    # Flatten the labels to be 1D

# Flatten images to 1D vectors
X_train = X_train.reshape(X_train.shape[0], -1)  # Shape: (num_samples, 28*28)
X_test = X_test.reshape(X_test.shape[0], -1)    # Shape: (num_samples, 28*28)

# Standardize the features (zero mean, unit variance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define a Softmax Regression model
softmax_model = LogisticRegression(
    solver="saga",              # Supports multinomial logistic regression
    max_iter=5000,              # Increase iterations if it doesn't converge
    class_weight="balanced",    # Handle imbalanced datasets
)

# Create a pipeline for PCA and Softmax Regression
pipeline = Pipeline([
    ("pca", PCA(n_components=0.95)),  # Retain 95% variance
    ("softmax", softmax_model)  # Step 2: Train Softmax Regression
])

# Define the parameter grid for GridSearchCV, experimented with 100 values but that takes too long
param_grid = {
    "softmax__C": ((10**np.linspace(10, -2, 20) * 0.5).tolist())
}

# Set up GridSearchCV
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,               # 5-fold cross-validation
    scoring="accuracy", # Optimize for accuracy
    verbose=2,          # Display progress
    n_jobs=-1           # Use all available CPU cores
)

# Perform the grid search
grid_search.fit(X_train, y_train)

# Get the best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("\nBest Hyperparameters:")
print(f"Regularization Strength (C): {best_params['softmax__C']}")

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)

# Print evaluation metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[str(i) for i in np.unique(y_train)]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Set Accuracy: {accuracy * 100:.2f}%")

# Plot the Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[str(i) for i in np.unique(y_train)], yticklabels=[str(i) for i in np.unique(y_train)],)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
