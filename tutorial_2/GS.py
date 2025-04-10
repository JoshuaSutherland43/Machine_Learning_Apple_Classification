# --- Cell 1: Imports ---
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import sys

# Add custom utilities path
sys.path.append("..")
from source.utils import split  # Make sure this exists in your source/utils.py

# --- Cell 2: Data Loading ---
try:
    # These lines only work in Jupyter, so they are skipped in .py
    # %store -r X
    # %store -r Y 
    # %store -r df
    raise Exception("Skipping %store for script")
except:
    # Fallback: Load data directly
    file_path = r'C:\Users\lab_services_student\Downloads\Apple_classifaction\data\Detect-GD.xlsx'
    df = pd.read_excel(file_path)

    # Data processing (replicating GD notebook steps)
    wavenumbers = np.float64(df.columns[4:])
    wavelengths = (1 / wavenumbers) * 1e7
    df.columns.values[4:] = np.round(wavelengths, 3)
    df['Condition'] = df['Condition'].str.upper()

    X = df.iloc[:, 4:]
    Y = df['Condition']
    print("Loaded data directly from file")

print(f"\nData shape: {X.shape}")
print(f"Sample conditions:\n{Y.value_counts()}")

# --- Cell 3: Data Visualization ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x=Y)
plt.title("Class Distribution")

plt.subplot(1, 2, 2)
# Plot first 50 samples
for i in range(50):
    color = 'r' if Y.iloc[i] == 'S' else 'b'
    plt.plot(X.columns, X.iloc[i], color, alpha=0.1)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance")
plt.title("Sample Spectra")
plt.tight_layout()
plt.show()

# --- Cell 4: Preprocessing ---
# Convert labels to binary
Y = Y.map({'S': 1, 'B': 0})

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# --- Cell 5: Train-Test Split ---
Xtrain, Xtest, Ytrain, Ytest = split(X_scaled, Y)

print("\nSplit Results:")
print(f"Training set: {Xtrain.shape[0]} samples")
print(f"Test set: {Xtest.shape[0]} samples")
print(f"\nClass balance in training:\n{Ytrain.value_counts()}")

# --- Cell 6: Model Training ---
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(Xtrain, Ytrain)

# --- Cell 7: Evaluation ---
# Predictions
train_pred = model.predict(Xtrain)
test_pred = model.predict(Xtest)

# Metrics
print("\nTraining Results:")
print(f"Accuracy: {accuracy_score(Ytrain, train_pred):.3f}")
print(f"Precision: {precision_score(Ytrain, train_pred):.3f}")

print("\nTest Results:")
print(f"Accuracy: {accuracy_score(Ytest, test_pred):.3f}")
print(f"Precision: {precision_score(Ytest, test_pred):.3f}")

# Confusion Matrix
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(Ytrain, train_pred), annot=True, fmt='d')
plt.title("Train Confusion Matrix")

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(Ytest, test_pred), annot=True, fmt='d')
plt.title("Test Confusion Matrix")
plt.tight_layout()
plt.show()
