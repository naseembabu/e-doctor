import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import numpy as np

df = pd.read_csv("heart_aug.csv")
df.info()
X = df.drop("target", axis=1)
from sklearn.preprocessing import MinMaxScaler# Scale the features to between -1 and 1
scaler=MinMaxScaler()
X=scaler.fit_transform(X)
y = df["target"]



# Assuming X and y are already defined
# X = ...
# y = ...

# Define models
models = {
    "XGBClassifier": XGBClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Naive Bayes": GaussianNB(),
    "K-Nearest Neighbour": KNeighborsClassifier(),
    "Adaboost": AdaBoostClassifier(),
    "Bagging": BaggingClassifier(),
    "MLP": MLPClassifier(max_iter=1000),
}

plt.figure(figsize=(10, 7))

# Define the cross-validation strategy
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, model in models.items():
    # Get cross-validated predictions
    y_pred = cross_val_predict(model, X, y, cv=cv)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    # Print evaluation results
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(classification_report(y, y_pred))
    print("\n")

    # Compute ROC curve and ROC area
    if hasattr(model, "predict_proba"):
        y_score = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
    else:  # Use decision function for models like SVC
        y_score = cross_val_predict(model, X, y, cv=cv, method='decision_function')

    fpr, tpr, _ = roc_curve(y, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10, prop={'weight': 'bold'})
plt.show()
