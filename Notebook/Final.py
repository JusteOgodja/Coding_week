<<<<<<< Updated upstream
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
=======



import numpy as np 
import pandas as pd 
>>>>>>> Stashed changes
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from scipy import stats
from imblearn.over_sampling import SMOTE

<<<<<<< Updated upstream
estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = fetch_ucirepo(id=544)

X = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.features
y = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.targets

file_path = '../Data/ObesityDataSet_raw_and_data_sinthetic.csv'
df = pd.read_csv(file_path)

df.head()

shape = df.shape
print(shape)

data_types = df.dtypes
data_types

missing_values = df.isnull().sum()
missing_values

summary_stats = df.describe(include='all')
summary_stats

sns.set_style("whitegrid")

plt.figure(figsize=(20, 10))

for i, col in enumerate(['Age', 'Height', 'Weight', 'FCVC','NCP', 'CH2O', 'FAF', 'TUE'], start=1):
    plt.subplot(2, 4, i)
    sns.histplot(df[col], bins=20, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 60))

=======

estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = fetch_ucirepo(id=544)


X = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.features
y = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.targets



file_path = '../Data/ObesityDataSet_raw_and_data_sinthetic.csv'
df = pd.read_csv(file_path)


df.head()



shape = df.shape
print(shape)





data_types = df.dtypes
data_types






missing_values = df.isnull().sum()
missing_values





summary_stats = df.describe(include='all')
summary_stats




plt.figure(figsize=(20, 60))


>>>>>>> Stashed changes
for i, col in enumerate(['Age', 'Height', 'Weight', 'FCVC','NCP', 'CH2O', 'FAF', 'TUE'], start=1):
    plt.subplot(8, 1, i)
    sns.boxplot(x='NObeyesdad', y=col, data=df)
    plt.title(f'Box Plot of {col} byNObeyesdad')
    plt.xlabel('NObeyesdad')
    plt.ylabel(col)

plt.tight_layout()
plt.show()

<<<<<<< Updated upstream
=======





>>>>>>> Stashed changes
def detect_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

Outliers = {}

for elt in ['Age', 'Height', 'Weight', 'FCVC','NCP', 'CH2O', 'FAF', 'TUE']:
<<<<<<< Updated upstream
    Outliers[elt] = detect_outliers(elt)
=======
  
    Outliers[elt] = detect_outliers(elt)



>>>>>>> Stashed changes
print(Outliers)

for colonne, outliers in Outliers.items():
    print(f"Outliers for {colonne}:")
    print(outliers)
    print("\n")  

<<<<<<< Updated upstream
for colonne, outliers in Outliers.items():
    print(f"Descriptive statistics for outliers in {colonne}:")
    print(outliers.describe(include='all'))
    print("\n") 
=======


for colonne, outliers in Outliers.items():
    print(f"Descriptive statistics for outliers in {colonne}:")
    print(outliers.describe(include='all'))
    print("\n")  
>>>>>>> Stashed changes

df = df[df["Age"] <= 50]
df

<<<<<<< Updated upstream
Outliers['Height'].shape

Outliers['Weight'].shape

Outliers['FCVC'].shape

Outliers['NCP'].shape

Outliers['CH2O'].shape

Outliers['FAF'].shape

Outliers['TUE'].shape

plt.figure(figsize=(20, 60))

=======




Outliers['Height'].shape




Outliers['Weight'].shape




Outliers['FCVC'].shape





Outliers['NCP'].shape





Outliers['CH2O'].shape




Outliers['FAF'].shape





Outliers['TUE'].shape




plt.figure(figsize=(20, 60))

e
>>>>>>> Stashed changes
for i, col in enumerate(['Gender', 'family_history_with_overweight', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad'], start=1):
    plt.subplot(8, 1, i)
    sns.countplot(x=col, data=df, palette='Set2')
    plt.title(f'Count Plot of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')

plt.tight_layout()
plt.show()

<<<<<<< Updated upstream
from sklearn.preprocessing import LabelEncoder

categorical_columns = ['Gender', 'family_history_with_overweight', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad', 'FAVC']

encoding_mappings = {}

=======



from sklearn.preprocessing import LabelEncoder


categorical_columns = ['Gender', 'family_history_with_overweight', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad', 'FAVC']


encoding_mappings = {}


>>>>>>> Stashed changes
for col in categorical_columns:
    le = LabelEncoder()
    df[f'{col}'] = le.fit_transform(df[col])
    encoding_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

<<<<<<< Updated upstream
for col, mapping in encoding_mappings.items():
    print(f"Correspondance pour {col} : {mapping}")
df

correlation_matrix = df.corr()

plt.figure(figsize=(25, 8))

=======

for col, mapping in encoding_mappings.items():
    print(f"Correspondance pour {col} : {mapping}")




df



correlation_matrix = df.corr()


plt.figure(figsize=(25, 8))


>>>>>>> Stashed changes
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=1, linecolor='black')
plt.title('Correlation Matrix of Numerical Variables')
plt.show()

<<<<<<< Updated upstream
cols_to_drop = ['FAVC', 'FCVC', 'SMOKE', 'NCP', 'CH2O', 'SCC', 'TUE']
df = df.drop(columns=cols_to_drop)

df.to_csv("../Data/Data_for_trainning.csv", index=False)
colonnes = df.columns.tolist()

X = df.drop(['NObeyesdad'], axis=1)
y = df['NObeyesdad']
=======




cols_to_drop = ['FAVC', 'FCVC', 'SMOKE', 'NCP', 'CH2O', 'SCC', 'TUE']
df = df.drop(columns=cols_to_drop)




df.to_csv("../Data/Data_for_trainning.csv", index=False)





colonnes = df.columns.tolist()




X = df.drop(['NObeyesdad'], axis=1)
y = df['NObeyesdad']


>>>>>>> Stashed changes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


df_resampled = pd.concat([X_resampled, y_resampled], axis=1)




df_resampled.to_csv("../Data/SMOTE_for_Data.csv", index=False)




print("Original Data:")
print(df.describe(include='all'))
print("\nResampled Data:")
print(df_resampled.describe(include='all'))


sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution of Numeric Variables in Resampled Data')

sns.histplot(df_resampled['Age'], kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Age Distribution')

sns.histplot(df_resampled['Height'], kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Height Distribution')

sns.histplot(df_resampled['Weight'], kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Weight Distribution')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Distribution of Categorical Variables in Resampled Data')

sns.countplot(data=df_resampled, x='Gender', ax=axes[0])
axes[0].set_title('Gender Distribution')

sns.countplot(data=df_resampled, x='NObeyesdad', ax=axes[1])
axes[1].set_title('Label Distribution')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

<<<<<<< Updated upstream
X = df_resampled.drop(['NObeyesdad'], axis=1)
y = df_resampled['NObeyesdad']

from sklearn.preprocessing import StandardScaler

=======

rbé la forme globale mais a pu ajouter des points synthétiques, rendant la distribution plus lisse.




X = df_resampled.drop(['NObeyesdad'], axis=1)
y = df_resampled['NObeyesdad']





from sklearn.preprocessing import StandardScaler




>>>>>>> Stashed changes
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

<<<<<<< Updated upstream
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'])
=======



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)





X_train.shape, X_test.shape, y_train.shape, y_test.shape






metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'])





>>>>>>> Stashed changes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
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

metrics_df = pd.DataFrame(metrics_dict)
metrics_df





from sklearn.tree import DecisionTreeClassifier


decision_tree_model = DecisionTreeClassifier(random_state=42)


decision_tree_model.fit(X_train, y_train)


y_pred_decision_tree = decision_tree_model.predict(X_test)

y_prob_decision_tree = decision_tree_model.predict_proba(X_test)

metrics_dict['Model'].append('Decision Tree')
metrics_dict['Accuracy'].append(accuracy_score(y_test, y_pred_decision_tree))
metrics_dict['Precision'].append(precision_score(y_test, y_pred_decision_tree, average='weighted'))
metrics_dict['Recall'].append(recall_score(y_test, y_pred_decision_tree, average='weighted'))
metrics_dict['F1-Score'].append(f1_score(y_test, y_pred_decision_tree, average='weighted'))
metrics_dict['AUC-ROC'].append(roc_auc_score(y_test, y_prob_decision_tree, multi_class='ovr', average='weighted'))


metrics_df = pd.DataFrame(metrics_dict)

metrics_df





from sklearn.ensemble import RandomForestClassifier

random_forest_model = RandomForestClassifier(random_state=42, n_estimators=100)


random_forest_model.fit(X_train, y_train)

y_pred_random_forest = random_forest_model.predict(X_test)


y_prob_random_forest = random_forest_model.predict_proba(X_test)


metrics_dict['Model'].append('Random Forest')
metrics_dict['Accuracy'].append(accuracy_score(y_test, y_pred_random_forest))
metrics_dict['Precision'].append(precision_score(y_test, y_pred_random_forest, average='weighted'))
metrics_dict['Recall'].append(recall_score(y_test, y_pred_random_forest, average='weighted'))
metrics_dict['F1-Score'].append(f1_score(y_test, y_pred_random_forest, average='weighted'))
metrics_dict['AUC-ROC'].append(roc_auc_score(y_test, y_prob_random_forest, multi_class='ovr', average='weighted'))


metrics_df = pd.DataFrame(metrics_dict)

metrics_df_sorted = metrics_df.sort_values(by='AUC-ROC', ascending=False)

metrics_df_sorted





from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV


svm_model = SVC(random_state=42, probability=True)


svm_model.fit(X_train, y_train)


y_pred_svm = svm_model.predict(X_test)


y_prob_svm = svm_model.predict_proba(X_test)


metrics_dict['Model'].append('SVM')
metrics_dict['Accuracy'].append(accuracy_score(y_test, y_pred_svm))
metrics_dict['Precision'].append(precision_score(y_test, y_pred_svm, average='weighted'))
metrics_dict['Recall'].append(recall_score(y_test, y_pred_svm, average='weighted'))
metrics_dict['F1-Score'].append(f1_score(y_test, y_pred_svm, average='weighted'))
metrics_dict['AUC-ROC'].append(roc_auc_score(y_test, y_prob_svm, multi_class='ovr', average='weighted'))


metrics_df = pd.DataFrame(metrics_dict)


metrics_df_sorted = metrics_df.sort_values(by='AUC-ROC', ascending=False)

metrics_df_sorted







from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)

y_prob_knn = knn_model.predict_proba(X_test)


metrics_dict['Model'].append('k-NN')
metrics_dict['Accuracy'].append(accuracy_score(y_test, y_pred_knn))
metrics_dict['Precision'].append(precision_score(y_test, y_pred_knn, average='weighted'))
metrics_dict['Recall'].append(recall_score(y_test, y_pred_knn, average='weighted'))
metrics_dict['F1-Score'].append(f1_score(y_test, y_pred_knn, average='weighted'))
metrics_dict['AUC-ROC'].append(roc_auc_score(y_test, y_prob_knn, multi_class='ovr', average='weighted'))


metrics_df = pd.DataFrame(metrics_dict)

r
metrics_df_sorted = metrics_df.sort_values(by='AUC-ROC', ascending=False)

metrics_df_sorted




from sklearn.naive_bayes import GaussianNB


naive_bayes_model = GaussianNB()


naive_bayes_model.fit(X_train, y_train)

y_pred_naive_bayes = naive_bayes_model.predict(X_test)


y_prob_naive_bayes = naive_bayes_model.predict_proba(X_test)


metrics_dict['Model'].append('Naive Bayes')
metrics_dict['Accuracy'].append(accuracy_score(y_test, y_pred_naive_bayes))
metrics_dict['Precision'].append(precision_score(y_test, y_pred_naive_bayes, average='weighted'))
metrics_dict['Recall'].append(recall_score(y_test, y_pred_naive_bayes, average='weighted'))
metrics_dict['F1-Score'].append(f1_score(y_test, y_pred_naive_bayes, average='weighted'))
metrics_dict['AUC-ROC'].append(roc_auc_score(y_test, y_prob_naive_bayes, multi_class='ovr', average='weighted'))


metrics_df = pd.DataFrame(metrics_dict)

metrics_df_sorted = metrics_df.sort_values(by='AUC-ROC', ascending=False)

metrics_df_sorted




from sklearn.ensemble import GradientBoostingClassifier


gradient_boosting_model = GradientBoostingClassifier(random_state=42)

gradient_boosting_model.fit(X_train, y_train)


y_pred_gradient_boosting = gradient_boosting_model.predict(X_test)

y_prob_gradient_boosting = gradient_boosting_model.predict_proba(X_test)

metrics_dict['Model'].append('Gradient Boosting')
metrics_dict['Accuracy'].append(accuracy_score(y_test, y_pred_gradient_boosting))
metrics_dict['Precision'].append(precision_score(y_test, y_pred_gradient_boosting, average='weighted'))
metrics_dict['Recall'].append(recall_score(y_test, y_pred_gradient_boosting, average='weighted'))
metrics_dict['F1-Score'].append(f1_score(y_test, y_pred_gradient_boosting, average='weighted'))
metrics_dict['AUC-ROC'].append(roc_auc_score(y_test, y_prob_gradient_boosting, multi_class='ovr', average='weighted'))

metrics_df = pd.DataFrame(metrics_dict)


metrics_df_sorted = metrics_df.sort_values(by='AUC-ROC', ascending=False)

metrics_df_sorted



t
xgboost_model = XGBClassifier(eval_metric='mlogloss')


xgboost_model.fit(X_train, y_train)


y_pred_xgboost = xgboost_model.predict(X_test)


y_prob_xgboost = xgboost_model.predict_proba(X_test)


metrics_dict['Model'].append('XGBoost')
metrics_dict['Accuracy'].append(accuracy_score(y_test, y_pred_xgboost))
metrics_dict['Precision'].append(precision_score(y_test, y_pred_xgboost, average='weighted'))
metrics_dict['Recall'].append(recall_score(y_test, y_pred_xgboost, average='weighted'))
metrics_dict['F1-Score'].append(f1_score(y_test, y_pred_xgboost, average='weighted'))
metrics_dict['AUC-ROC'].append(roc_auc_score(y_test, y_prob_xgboost, multi_class='ovr', average='weighted'))


metrics_df = pd.DataFrame(metrics_dict)
metrics_df_sorted = metrics_df.sort_values(by='AUC-ROC', ascending=False)
metrics_df_sorted





from lightgbm import LGBMClassifier


lightgbm_model = LGBMClassifier(
    n_estimators=100,      
    max_depth=7,           
    min_data_in_leaf=20,   
    min_split_gain=0.01,   
    class_weight='balanced' 
)



lightgbm_model.fit(X_train, y_train)


y_pred_lightgbm = lightgbm_model.predict(X_test)


y_prob_lightgbm = lightgbm_model.predict_proba(X_test)


metrics_dict['Model'].append('LightGBM')
metrics_dict['Accuracy'].append(accuracy_score(y_test, y_pred_lightgbm))
metrics_dict['Precision'].append(precision_score(y_test, y_pred_lightgbm, average='weighted'))
metrics_dict['Recall'].append(recall_score(y_test, y_pred_lightgbm, average='weighted'))
metrics_dict['F1-Score'].append(f1_score(y_test, y_pred_lightgbm, average='weighted'))
metrics_dict['AUC-ROC'].append(roc_auc_score(y_test, y_prob_lightgbm, multi_class='ovr', average='weighted'))


metrics_df = pd.DataFrame(metrics_dict)
metrics_df_sorted = metrics_df.sort_values(by='AUC-ROC', ascending=False)
metrics_df_sorted





from catboost import CatBoostClassifier



catboost_model.fit(X_train, y_train)


y_pred_catboost = catboost_model.predict(X_test)

y_prob_catboost = catboost_model.predict_proba(X_test)

metrics_dict['Model'].append('CatBoost')
metrics_dict['Accuracy'].append(accuracy_score(y_test, y_pred_catboost))
metrics_dict['Precision'].append(precision_score(y_test, y_pred_catboost, average='weighted'))
metrics_dict['Recall'].append(recall_score(y_test, y_pred_catboost, average='weighted'))
metrics_dict['F1-Score'].append(f1_score(y_test, y_pred_catboost, average='weighted'))
metrics_dict['AUC-ROC'].append(roc_auc_score(y_test, y_prob_catboost, multi_class='ovr', average='weighted'))


metrics_df = pd.DataFrame(metrics_dict)
metrics_df_sorted = metrics_df.sort_values(by='AUC-ROC', ascending=False)
metrics_df_sorted





def optimize_memory(df):
    print(f"Memory usage before optimization: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    
    for column in df.columns:
        col_type = df[column].dtype
        
        if np.issubdtype(col_type, np.number):
            # Si la colonne est de type float64, la convertir en float32 si possible
            if col_type == np.float64:
                df[column] = df[column].astype(np.float32)
            # Si la colonne est de type int64, la convertir en int32 ou int16 si possible
            elif col_type == np.int64:
                if df[column].min() >= np.iinfo(np.int32).min and df[column].max() <= np.iinfo(np.int32).max:
                    df[column] = df[column].astype(np.int32)
                elif df[column].min() >= np.iinfo(np.int16).min and df[column].max() <= np.iinfo(np.int16).max:
                    df[column] = df[column].astype(np.int16)
        
        # Optimiser les colonnes de type catégorie
        elif df[column].dtype == 'object':
            # Convertir les chaînes de caractères en catégorie si possible
            if df[column].nunique() / len(df[column]) < 0.5:
                df[column] = df[column].astype('category')
    
    # Afficher la mémoire utilisée après l'optimisation
    print(f"Memory usage after optimization: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df


# In[52]:


# Vérifier l'utilisation mémoire avant optimisation
print("Before Optimization:")
print(df_resampled.memory_usage(deep=True))

# Optimiser la mémoire
df_optimized = optimize_memory(df_resampled)

# Vérifier l'utilisation mémoire après optimisation
print("\nAfter Optimization:")
print(df_optimized.memory_usage(deep=True))


# **Avant optimisation :**  
# - **Mémoire totale** : **0.21 MB** pour **19,600 lignes**.  
# - Les colonnes numériques étaient stockées **float64** — ce qui prend plus de place qu’il n’en faut.  
# - Les colonnes textuelles (comme les catégories) étaient au format **object**, ce qui utilise beaucoup de mémoire pour stocker les chaînes de caractères.  
# 
# 
# **Optimisations appliquées :**  
# 
# 1. **Conversion des flottants** :  
#    - Les **float64** ont été convertis en **float32** — cela réduit de **50%** la mémoire utilisée par ces colonnes.  
#    - Utilisation : parfait pour des variables comme **Weight**, **Height** et **BMI** où une précision à la virgule est nécessaire, mais **float64** est souvent surdimensionné.
# 
# 2. **Conversion des colonnes catégoriques** :  
#    - Les colonnes de type **object** (comme **family_history_with_overweight** ou **MTRANS**) ont été converties en **category**.  
#    - Gain : cela réduit la mémoire en encodant chaque catégorie comme un entier au lieu de stocker chaque chaîne de caractères, surtout si le nombre de catégories est faible comparé à la taille totale du dataset.  
# 
# 
# **Après optimisation :**  
# - **Mémoire totale** : **0.10 MB** — **réduction de 52%**.  
# - **Nombre de lignes** : réduit à **9,800** 
# - Les colonnes numériques et catégoriques utilisent désormais le type de données le plus léger possible sans perte de précision.
# 
# 
# 
# 

# In[53]:


# Comparaison visuelle de la mémoire avant et après optimisation
memory_before = df.memory_usage(deep=True).sum() / 1024**2
memory_after = df_optimized.memory_usage(deep=True).sum() / 1024**2

# Création d'un graphique
plt.bar(['Before', 'After'], [memory_before, memory_after], color=['red', 'green'])
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage Before and After Optimization')
plt.show()


# ## SHAP Explainability

# Comprendre le fonctionnement interne d'un modèle d'apprentissage automatique est fondamental pour en garantir la fiabilité. Grâce aux valeurs SHAP (SHapley Additive exPlanations), nous pouvons expliquer l'impact de chaque variable sur les prédictions du modèle. Cette transparence nous permet d'identifier les caractéristiques les plus influentes et d'ajuster nos approches en conséquence.

# In[54]:


import shap
import matplotlib.pyplot as plt
import pandas as pd  # Assurez-vous d'importer Pandas

# Création de l'explainer SHAP pour CatBoost
explainer = shap.Explainer(catboost_model)

# Calcul des valeurs SHAP pour les données de test
shap_values = explainer(X_test)

# 1. Affichage du résumé SHAP (summary plot)
shap.summary_plot(shap_values, X_test)

# 2. Affichage de l'importance des features (bar plot)
shap.summary_plot(shap_values, X_test, plot_type="bar")

# 3. Analyse d'un échantillon spécifique (force plot)
shap.initjs()


# **Interprétation globale des résultats :**
# 
# 1. **Importance des caractéristiques (SHAP values)**  
#    - **Feature 9** a le plus grand impact sur les prédictions du modèle, avec une contribution significative pour toutes les classes.
#    - **Feature 3** et **Feature 0** suivent avec une importance notable, bien qu'elles aient un impact globalement moindre que Feature 9.
#    - Les autres caractéristiques (Feature 5, 8, 7, etc.) ont un impact marginal, ce qui signifie qu'elles influencent peu les prédictions du modèle.
# 
# 2. **Interprétation par classe**  
#    - Chaque barre colorée montre l’impact moyen des variables sur les prédictions pour chaque classe (de Class 0 à Class 6).  
#    - Par exemple, **Feature 9** joue un rôle crucial dans toutes les classes, tandis que certaines caractéristiques comme **Feature 5** n'ont presque aucun effet, quel que soit le groupe.
# 
# 

# In[55]:


import joblib


# In[56]:


filename = '../Data/catboost_model.sav'
joblib.dump(random_forest_model, filename)


# In[ ]:




