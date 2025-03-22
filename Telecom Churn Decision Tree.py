# Import Lib
import numpy as np
import pandas as pd

# Load CSV
train_df = pd.read_csv('churn-bigml-80.csv')
test_df = pd.read_csv('churn-bigml-20.csv')
# print(train_df.head())
# print(test_df.head())
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values
# print(X_test[:5])
# print(y_test[:50])

# Label encoder
from sklearn.preprocessing import LabelEncoder
Le = LabelEncoder()
X_train[:, 0] = Le.fit_transform(X_train[:, 0])
X_train[:, 1] = Le.fit_transform(X_train[:, 1])
y_train = Le.fit_transform(y_train)
# print(X_train[:5])
# print(y_train[:50])
X_test[:, 0] = Le.fit_transform(X_test[:, 0])
X_test[:, 1] = Le.fit_transform(X_test[:, 1])
y_test = Le.fit_transform(y_test)
# print(X_test[:10])
# print(y_test[:50])

# # One hot encoder for coulumns 0
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# # ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['State'])], remainder='passthrough')
# X_train = ct.fit_transform(X_train).toarray()
# X_test = ct.fit_transform(X_test).toarray()
# print(X_train[:1])
# # print(X_test[:1])

# Standard Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# print(X_train[:3])
# print(X_test[:3])

# Train the model
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Compare the Train Output to Predict Output
y_pred = dt_model.predict(X_test)
print(np.concatenate((y_pred[:100].reshape(100,1), y_test[:100].reshape(100,1)), axis=1))


import joblib
# Save the trained model
joblib.dump(dt_model, "churn_model.pkl")
# Save the StandardScaler and OneHotEncoder
joblib.dump(sc, "scaler.pkl")
# joblib.dump(ct, "column_transformer.pkl")
# Load the trained model
# model = joblib.load("churn_model.pkl")
# # Load the StandardScaler and OneHotEncoder
# scaler = joblib.load("scaler.pkl")
# column_transformer = joblib.load("column_transformer.pkl")
# # Predict using the loaded model
# y_pred = model.predict(X_test)

# Accuracy Score
from sklearn.metrics import accuracy_score
y_pred = dt_model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
# //Accuracy: 0.9505

# F1 Score 
from sklearn.metrics import f1_score
y_pred = dt_model.predict(X_test)
print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
# //F1-score: 0.8216

# Recall Score
from sklearn.metrics import recall_score
y_pred = dt_model.predict(X_test)
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
# //Recall: 0.8000

# Precision Score
from sklearn.metrics import precision_score
y_pred = dt_model.predict(X_test)
print(f"Precision: {precision_score(y_test, y_pred):.4f}\n")
# //Precision: 0.8444