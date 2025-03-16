from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_and_evaluate_models(X, y, test_size=0.3, random_state=42):
    
    # Standardizing features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Splitting dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=random_state, solver='lbfgs'),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "SVM": SVC(random_state=42, probability=True),
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric='mlogloss'),
        "LightGBM": LGBMClassifier(n_estimators=100, max_depth=7, min_data_in_leaf=20, min_split_gain=0.01, class_weight='balanced'),
        "CatBoost": CatBoostClassifier(silent=True)
    }
    
    metrics_dict = {"Model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1-Score": [], "AUC-ROC": []}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        metrics_dict["Model"].append(name)
        metrics_dict["Accuracy"].append(accuracy_score(y_test, y_pred))
        metrics_dict["Precision"].append(precision_score(y_test, y_pred, average='weighted'))
        metrics_dict["Recall"].append(recall_score(y_test, y_pred, average='weighted'))
        metrics_dict["F1-Score"].append(f1_score(y_test, y_pred, average='weighted'))
        
        if y_prob is not None:
            metrics_dict["AUC-ROC"].append(roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted'))
        else:
            metrics_dict["AUC-ROC"].append(None)
    
    return metrics_dict
