import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score

file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
data = pd.read_csv(file_path)
data.head()

print(data)

"""# Distribution of target variable
sns.countplot(y='NObeyesdad', data=data)
plt.title('Distribution of Obesity Levels')
plt.show()"""


# Standardizing continuous numerical features
continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[continuous_columns])


# Converting to a DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))


# Combining with the original dataset
scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)

# Identifying categorical columns
categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('NObeyesdad')  # Exclude target column

# Applying one-hot encoding
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])

# Converting to a DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
print(encoded_df)
# Combining with the original dataset
prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)

# Encoding the target variable
prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes
prepped_data.head()


# Preparing final dataset
X = prepped_data.drop('NObeyesdad', axis=1)
y = prepped_data['NObeyesdad']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Training logistic regression model using One-vs-All (default)
model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ova.fit(X_train, y_train)

# Predictions
y_pred_ova = model_ova.predict(X_test)

# Evaluation metrics for OvA
print("One-vs-All (OvA) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")


#Logistic Regression Accuracy with Different Test Sizes

"""for size in [0.1,0.2,0.3]:
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=size, random_state=42, stratify=y
    )

    # Initialize and train the model
    model = LogisticRegression(multi_class='ovr', max_iter=1000)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test size: {size:.1f} -> Accuracy: {accuracy*100:.2f}%")"""

#Bar chart of feature importance using the coefficients from the One vs All logistic regression and One vs One model.

# Get average absolute coefficients across all OvA classifiers
avg_importance_ova = np.mean(np.abs(model_ova.coef_), axis=0)

# Get feature names
feature_names = X_train.columns

# Create a bar chart
plt.figure(figsize=(12, 6))
plt.barh(feature_names, avg_importance_ova)
plt.title("Feature Importance (OvA Logistic Regression)")
plt.xlabel("Average |Coefficient| across classes")
plt.tight_layout()
plt.show()


# Wrap logistic regression in One-vs-One strategy
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(X_train, y_train)

y_pred_ovo = model_ovo.predict(X_test)

# Evaluation metrics for OvA
print("One-vs-One Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")

# Each classifier compares two classes
# Average absolute importance across all binary classifiers
all_coefs_ovo = np.abs(np.vstack([est.coef_[0] for est in model_ovo.estimators_]))
avg_importance_ovo = np.mean(all_coefs_ovo, axis=0)

# Bar chart
plt.figure(figsize=(12, 6))
plt.barh(feature_names, avg_importance_ovo)
plt.title("Feature Importance (OvO Logistic Regression)")
plt.xlabel("Average |Coefficient| across binary classifiers")
plt.tight_layout()
plt.show()

