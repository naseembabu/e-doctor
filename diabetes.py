import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
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
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
# Assuming dia_df is your DataFrame
# dia_df = pd.read_csv('path_to_your_data.csv')  # Uncomment and modify if you need to load the data
dia_df = pd.read_csv("/content/dia01_aug_updated - Sheet1.csv")
dia_df['Outcome'].value_counts()
print(dia_df['Outcome'].value_counts())
# Display the first 10 rows of the DataFrame to check for any obvious issues
print(dia_df.head(10))

# Check for missing values in the DataFrame
missing_values = dia_df.isnull().sum()
print("Missing values in each column:")
print(missing_values)

# Fill missing values if any
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(dia_df.drop(columns='Outcome'))
y = dia_df['Outcome']
# Standardize the features
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# X = scaler.transform(X)

from sklearn.preprocessing import MinMaxScaler# Scale the features to between -1 and 1
scaler=MinMaxScaler()
X=scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


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

