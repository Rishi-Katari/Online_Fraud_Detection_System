import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest Classifier
from sklearn.metrics import accuracy_score
import pickle

# Load the data
data = pd.read_csv("creditcard.csv")

# Display data information
data.head()
data.info()
data.describe()

# Check for missing values
print("Missing values in each column before filling:")
print(data.isna().sum())

# Fill NaN values with the mean of their respective columns
data.fillna(data.mean(), inplace=True)

# Check for missing values again
print("Missing values in each column after filling:")
print(data.isna().sum())

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Display counts
print("Legitimate transactions count:")
print(legit.value_counts())

print("Fraud transactions count:")
print(fraud.value_counts())

# Display class distribution
data['Class'].value_counts()

# Resampling legitimate transactions
legit_samples = legit.sample(n=492, random_state=42)  # Ensure reproducibility
new_data = pd.concat([legit_samples, fraud], axis=0)

# Check new class distribution
print("New data class distribution:")
print(new_data["Class"].value_counts())

# Features and target variable
X = new_data.drop(columns="Class", axis=1)
y = new_data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)  # Optional: Set a random seed for reproducibility
model.fit(X_train, y_train)

# Predict on training data
x_pred_train = model.predict(X_train)

# Calculate training accuracy
acc_score_train = accuracy_score(x_pred_train, y_train)
print(f"Training accuracy: {acc_score_train}")

# Predict on test data
x_pred_test = model.predict(X_test)

# Calculate test accuracy
acc_score_test = accuracy_score(x_pred_test, y_test)
print(f"Test accuracy: {acc_score_test}")

# Save the model using pickle
pickle.dump(model, open("model.pkl", "wb"))
