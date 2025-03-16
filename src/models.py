from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Fonction pour la préparation des données et la création du DataFrame vide
def prepare_data(X_scaled, y):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'])
    return X_train, X_test, y_train, y_test, metrics_df

# Fonction pour Logistic Regression
def logistic_regression(X_train, X_test, y_train, y_test, metrics_df):
    logistic_model = LogisticRegression(random_state=42, solver='lbfgs')
    logistic_model.fit(X_train, y_train)
    
    y_pred_logistic = logistic_model.predict(X_test)
    y_prob_logistic = logistic_model.predict_proba(X_test)

    metrics_dict = {
        'Model': ['Logistic Regression'],
        'Accuracy': [accuracy_score(y_test, y_pred_logistic)],
        'Precision': [precision_score(y_test, y_pred_logistic, average='weighted')],
        'Recall': [recall_score(y_test, y_pred_logistic, average='weighted')],
        'F1-Score': [f1_score(y_test, y_pred_logistic, average='weighted')],
        'AUC-ROC': [roc_auc_score(y_test, y_prob_logistic, multi_class='ovr', average='weighted')]
    }

    metrics_df = metrics_df.append(pd.DataFrame(metrics_dict), ignore_index=True)
    return metrics_df

# Fonction pour Decision Tree
def decision_tree(X_train, X_test, y_train, y_test, metrics_df):
    decision_tree_model = DecisionTreeClassifier(random_state=42)
    decision_tree_model.fit(X_train, y_train)
    
    y_pred_decision_tree = decision_tree_model.predict(X_test)
    y_prob_decision_tree = decision_tree_model.predict_proba(X_test)
    
    metrics_dict = {
        'Model': ['Decision Tree'],
        'Accuracy': [accuracy_score(y_test, y_pred_decision_tree)],
        'Precision': [precision_score(y_test, y_pred_decision_tree, average='weighted')],
        'Recall': [recall_score(y_test, y_pred_decision_tree, average='weighted')],
        'F1-Score': [f1_score(y_test, y_pred_decision_tree, average='weighted')],
        'AUC-ROC': [roc_auc_score(y_test, y_prob_decision_tree, multi_class='ovr', average='weighted')]
    }

    metrics_df = metrics_df.append(pd.DataFrame(metrics_dict), ignore_index=True)
    return metrics_df

# Fonction pour Random Forest
def random_forest(X_train, X_test, y_train, y_test, metrics_df):
    random_forest_model = RandomForestClassifier(random_state=42, n_estimators=100)
    random_forest_model.fit(X_train, y_train)

    y_pred_random_forest = random_forest_model.predict(X_test)
    y_prob_random_forest = random_forest_model.predict_proba(X_test)

    metrics_dict = {
        'Model': ['Random Forest'],
        'Accuracy': [accuracy_score(y_test, y_pred_random_forest)],
        'Precision': [precision_score(y_test, y_pred_random_forest, average='weighted')],
        'Recall': [recall_score(y_test, y_pred_random_forest, average='weighted')],
        'F1-Score': [f1_score(y_test, y_pred_random_forest, average='weighted')],
        'AUC-ROC': [roc_auc_score(y_test, y_prob_random_forest, multi_class='ovr', average='weighted')]
    }

    metrics_df = metrics_df.append(pd.DataFrame(metrics_dict), ignore_index=True)
    return metrics_df

# Fonction pour SVM
def svm(X_train, X_test, y_train, y_test, metrics_df):
    svm_model = SVC(random_state=42, probability=True)
    svm_model.fit(X_train, y_train)
    
    y_pred_svm = svm_model.predict(X_test)
    y_prob_svm = svm_model.predict_proba(X_test)

    metrics_dict = {
        'Model': ['SVM'],
        'Accuracy': [accuracy_score(y_test, y_pred_svm)],
        'Precision': [precision_score(y_test, y_pred_svm, average='weighted')],
        'Recall': [recall_score(y_test, y_pred_svm, average='weighted')],
        'F1-Score': [f1_score(y_test, y_pred_svm, average='weighted')],
        'AUC-ROC': [roc_auc_score(y_test, y_prob_svm, multi_class='ovr', average='weighted')]
    }

    metrics_df = metrics_df.append(pd.DataFrame(metrics_dict), ignore_index=True)
    return metrics_df

# Fonction pour k-NN
def knn(X_train, X_test, y_train, y_test, metrics_df):
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    
    y_pred_knn = knn_model.predict(X_test)
    y_prob_knn = knn_model.predict_proba(X_test)

    metrics_dict = {
        'Model': ['k-NN'],
        'Accuracy': [accuracy_score(y_test, y_pred_knn)],
        'Precision': [precision_score(y_test, y_pred_knn, average='weighted')],
        'Recall': [recall_score(y_test, y_pred_knn, average='weighted')],
        'F1-Score': [f1_score(y_test, y_pred_knn, average='weighted')],
        'AUC-ROC': [roc_auc_score(y_test, y_prob_knn, multi_class='ovr', average='weighted')]
    }

    metrics_df = metrics_df.append(pd.DataFrame(metrics_dict), ignore_index=True)
    return metrics_df

# Fonction pour Naive Bayes
def naive_bayes(X_train, X_test, y_train, y_test, metrics_df):
    naive_bayes_model = GaussianNB()
    naive_bayes_model.fit(X_train, y_train)

    y_pred_naive_bayes = naive_bayes_model.predict(X_test)
    y_prob_naive_bayes = naive_bayes_model.predict_proba(X_test)

    metrics_dict = {
        'Model': ['Naive Bayes'],
        'Accuracy': [accuracy_score(y_test, y_pred_naive_bayes)],
        'Precision': [precision_score(y_test, y_pred_naive_bayes, average='weighted')],
        'Recall': [recall_score(y_test, y_pred_naive_bayes, average='weighted')],
        'F1-Score': [f1_score(y_test, y_pred_naive_bayes, average='weighted')],
        'AUC-ROC': [roc_auc_score(y_test, y_prob_naive_bayes, multi_class='ovr', average='weighted')]
    }

    metrics_df = metrics_df.append(pd.DataFrame(metrics_dict), ignore_index=True)
    return metrics_df

# Fonction pour Gradient Boosting
def gradient_boosting(X_train, X_test, y_train, y_test, metrics_df):
    gradient_boosting_model = GradientBoostingClassifier(random_state=42)
    gradient_boosting_model.fit(X_train, y_train)

    y_pred_gradient_boosting = gradient_boosting_model.predict(X_test)
    y_prob_gradient_boosting = gradient_boosting_model.predict_proba(X_test)

    metrics_dict = {
        'Model': ['Gradient Boosting'],
        'Accuracy': [accuracy_score(y_test, y_pred_gradient_boosting)],
        'Precision': [precision_score(y_test, y_pred_gradient_boosting, average='weighted')],
        'Recall': [recall_score(y_test, y_pred_gradient_boosting, average='weighted')],
        'F1-Score': [f1_score(y_test, y_pred_gradient_boosting, average='weighted')],
        'AUC-ROC': [roc_auc_score(y_test, y_prob_gradient_boosting, multi_class='ovr', average='weighted')]
    }

    metrics_df = metrics_df.append(pd.DataFrame(metrics_dict), ignore_index=True)
    return metrics_df

# Fonction pour XGBoost
def xgboost(X_train, X_test, y_train, y_test, metrics_df):
    xgboost_model = XGBClassifier(eval_metric='mlogloss')
    xgboost_model.fit(X_train, y_train)

    y_pred_xgboost = xgboost_model.predict(X_test)
    y_prob_xgboost = xgboost_model.predict_proba(X_test)

    metrics_dict = {
        'Model': ['XGBoost'],
        'Accuracy': [accuracy_score(y_test, y_pred_xgboost)],
        'Precision': [precision_score(y_test, y_pred_xgboost, average='weighted')],
        'Recall': [recall_score(y_test, y_pred_xgboost, average='weighted')],
        'F1-Score': [f1_score(y_test, y_pred_xgboost, average='weighted')],
        'AUC-ROC': [roc_auc_score(y_test, y_prob_xgboost, multi_class='ovr', average='weighted')]
    }

    metrics_df = metrics_df.append(pd.DataFrame(metrics_dict), ignore_index=True)
    return metrics_df

# Fonction pour LightGBM
def lightgbm(X_train, X_test, y_train, y_test, metrics_df):
    lightgbm_model = LGBMClassifier()
    lightgbm_model.fit(X_train, y_train)

    y_pred_lightgbm = lightgbm_model.predict(X_test)
    y_prob_lightgbm = lightgbm_model.predict_proba(X_test)

    metrics_dict = {
        'Model': ['LightGBM'],
        'Accuracy': [accuracy_score(y_test, y_pred_lightgbm)],
        'Precision': [precision_score(y_test, y_pred_lightgbm, average='weighted')],
        'Recall': [recall_score(y_test, y_pred_lightgbm, average='weighted')],
        'F1-Score': [f1_score(y_test, y_pred_lightgbm, average='weighted')],
        'AUC-ROC': [roc_auc_score(y_test, y_prob_lightgbm, multi_class='ovr', average='weighted')]
    }

    metrics_df = metrics_df.append(pd.DataFrame(metrics_dict), ignore_index=True)
    return metrics_df

# Fonction pour CatBoost
def catboost(X_train, X_test, y_train, y_test, metrics_df):
    catboost_model = CatBoostClassifier(verbose=0)
    catboost_model.fit(X_train, y_train)

    y_pred_catboost = catboost_model.predict(X_test)
    y_prob_catboost = catboost_model.predict_proba(X_test)

    metrics_dict = {
        'Model': ['CatBoost'],
        'Accuracy': [accuracy_score(y_test, y_pred_catboost)],
        'Precision': [precision_score(y_test, y_pred_catboost, average='weighted')],
        'Recall': [recall_score(y_test, y_pred_catboost, average='weighted')],
        'F1-Score': [f1_score(y_test, y_pred_catboost, average='weighted')],
        'AUC-ROC': [roc_auc_score(y_test, y_prob_catboost, multi_class='ovr', average='weighted')]
    }

    metrics_df = metrics_df.append(pd.DataFrame(metrics_dict), ignore_index=True)
    return metrics_df

# Fonction pour trier les résultats
def sort_metrics(metrics_df):
    metrics_df_sorted = metrics_df.sort_values(by='AUC-ROC', ascending=False)
    return metrics_df_sorted

# Exemple d'utilisation :
# X_train, X_test, y_train, y_test, metrics_df = prepare_data(X_scaled, y)
# metrics_df = logistic_regression(X_train, X_test, y_train, y_test, metrics_df)
# metrics_df = decision_tree(X_train, X_test, y_train, y_test, metrics_df)
# metrics_df = random_forest(X_train, X_test, y_train, y_test, metrics_df)
# metrics_df = svm(X_train, X_test, y_train, y_test, metrics_df)
# metrics_df = knn(X_train, X_test, y_train, y_test, metrics_df)
# metrics_df = naive_bayes(X_train, X_test, y_train, y_test, metrics_df)
# metrics_df = gradient_boosting(X_train, X_test, y_train, y_test, metrics_df)
# metrics_df = xgboost(X_train, X_test, y_train, y_test, metrics_df)
# metrics_df = lightgbm(X_train, X_test, y_train, y_test, metrics_df)
# metrics_df = catboost(X_train, X_test, y_train, y_test, metrics_df)
# metrics_df_sorted = sort_metrics(metrics_df)
