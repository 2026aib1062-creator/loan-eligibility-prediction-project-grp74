import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
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


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=100,
    stratify=y
#     model = LogisticRegression(max_iter=100)
# model.fit(X_train_scaled, y_train)

# print("Logistic Regression model trained successfully!")

)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(max_iter=100)
model.fit(X_train_scaled, y_train)

print("Logistic Regression model trained successfully!")

prediction = model.predict(X_test_scaled)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)



metric_names = ["Precision", "Recall", "F1-score"]
metric_values = [precision, recall, f1]

plt.figure(figsize=(8, 5))
bars = plt.bar(metric_names, metric_values)

plt.title("Logistic Regression: Performance Metric Comparison")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bar, score in zip(bars, metric_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{score:.3f}",
        ha="center"
    )

plt.tight_layout()
plt.show()
