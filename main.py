import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay

#Loading and Cleaning the code
df = pd.read_csv("Loan_Data.csv")
df = df.drop("Loan_ID", axis=1)
df = df.drop("Gender", axis=1)
df = df.dropna(subset=["Married","Loan_Amount_Term","Credit_History","Self_Employed","LoanAmount"])
df['Dependents'] = df['Dependents'].str.replace('+', '', regex=False)
df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce').astype('Int64')
df["Dependents"] = df["Dependents"].fillna(0)
df = pd.get_dummies(df, drop_first=True)
X = df.drop("Loan_Status_Y", axis=1)
y = df["Loan_Status_Y"]
print("Data loaded and cleaned.")
print("Shape of clean dataset:", df.shape)

#Splitting the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=100,
    stratify=y

)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(max_iter=100)
model.fit(X_train_scaled, y_train)
print("Logistic Regression model trained successfully!")

#Making predictions using the trained model and analysing the model
prediction = model.predict(X_test_scaled)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
print("Made predictions on test data. The following are the performance metrics for the trained logistic regression model.")
print("\nPrecision:", round(precision,3))
print("Recall:", round(recall,3))
print("F1-score:", round(f1,3))

#Making predictions on training data using the trained model
prediction_train = model.predict(X_train_scaled)
precision_train = precision_score(y_train, prediction_train)
recall_train = recall_score(y_train, prediction_train)
f1_train = f1_score(y_train, prediction_train)
print("\nMade predictions on training data. The following are the performance metrics for the trained logistic regression model.")
print("\nPrecision:", round(precision_train,3))
print("Recall:", round(recall_train,3))
print("F1-score:", round(f1_train,3))

#Graphing and saving the confusion matrix and the bar graph into the Outputs folder
actual_counts = y_test.value_counts().sort_index()

predicted_counts = pd.Series(prediction).value_counts().sort_index()

labels = ["Not Eligible (0)", "Eligible (1)"]
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(
    x - width / 2,
    actual_counts,
    width,
    label="Actual",
    color="skyblue",
    edgecolor="black",
)
ax.bar(
    x + width / 2,
    predicted_counts,
    width,
    label="Predicted",
    color="salmon",
    edgecolor="black",
)

ax.set_xlabel("Class Category", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title(
    "Comparison of Actual vs. Predicted Class Distributions",
    fontsize=14,
    weight="bold",
)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.tight_layout()
plt.savefig("Output/Loan Eligibility prediction comparison.png", dpi=300, bbox_inches="tight")
print("\nSaved prediction comparison; 'Output/Loan Eligibility prediction comparison.png'")
plt.show()

fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay.from_predictions(
    y_test, prediction, display_labels=["Not Eligible", "Eligible"], cmap="Blues", ax=ax
)
plt.title("Confusion Matrix", fontsize=14, weight="bold")

plt.savefig("Output/confusion_matrix.png", dpi=300, bbox_inches="tight")
print("Saved confusion matrix; 'Output/confusion_matrix.png'")
plt.show()