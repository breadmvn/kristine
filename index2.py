import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
train = pd.read_csv(r'train.csv')
test = pd.read_csv(r'train.csv')
train = train.fillna(train.median(numeric_only=True))
train = train.dropna(subset=["loan_status"])
train = pd.get_dummies(train, drop_first=True)
X = train.drop("loan_status", axis=1)
y = train["loan_status"].map({1: 1, 0: 0})
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))
test = test.fillna(test.median(numeric_only=True))
test = pd.get_dummies(test, drop_first=True)
test = test.reindex(columns=X.columns, fill_value=0)
test_predictions = model.predict(test)
print(test_predictions)