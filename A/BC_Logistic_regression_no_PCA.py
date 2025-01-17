

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns


# Load BreastMNIST dataset from the local .npz file
data_path = './Datasets/breastmnist.npz'  # Adjust path if necessary
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

# Standardising the features (zero mean, unit variance) for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression Model
logistic_model = LogisticRegression(max_iter=1000, class_weight='balanced')

# Generate learning curve data
train_sizes, train_scores, test_scores = learning_curve(
    logistic_model,
    X_train,
    y_train,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,  # Use all available cores
    train_sizes=np.linspace(0.1, 1.0, 10)  # Use 10 increments from 10% to 100% of the training data
)

# Compute the mean and standard deviation of the training and validation scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training Accuracy', color='blue', marker='o')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)
plt.plot(train_sizes, test_mean, label='Validation Accuracy', color='orange', marker='s')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='orange', alpha=0.2)
plt.title('Learning Curve for Logistic Regression')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)
plt.legend()
plt.grid()
plt.show()

# Train the model on the full training data
logistic_model.fit(X_train, y_train.ravel())

# Make Predictions
y_pred = logistic_model.predict(X_test)

# Evaluate Model
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")


# Plot the Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()



