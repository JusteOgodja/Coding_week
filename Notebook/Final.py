import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from scipy import stats
from imblearn.over_sampling import SMOTE

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

for i, col in enumerate(['Age', 'Height', 'Weight', 'FCVC','NCP', 'CH2O', 'FAF', 'TUE'], start=1):
    plt.subplot(8, 1, i)
    sns.boxplot(x='NObeyesdad', y=col, data=df)
    plt.title(f'Box Plot of {col} byNObeyesdad')
    plt.xlabel('NObeyesdad')
    plt.ylabel(col)

plt.tight_layout()
plt.show()

def detect_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

Outliers = {}

for elt in ['Age', 'Height', 'Weight', 'FCVC','NCP', 'CH2O', 'FAF', 'TUE']:
    Outliers[elt] = detect_outliers(elt)
print(Outliers)

for colonne, outliers in Outliers.items():
    print(f"Outliers for {colonne}:")
    print(outliers)
    print("\n")  

for colonne, outliers in Outliers.items():
    print(f"Descriptive statistics for outliers in {colonne}:")
    print(outliers.describe(include='all'))
    print("\n") 

df = df[df["Age"] <= 50]
df

Outliers['Height'].shape

Outliers['Weight'].shape

Outliers['FCVC'].shape

Outliers['NCP'].shape

Outliers['CH2O'].shape

Outliers['FAF'].shape

Outliers['TUE'].shape

plt.figure(figsize=(20, 60))

for i, col in enumerate(['Gender', 'family_history_with_overweight', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad'], start=1):
    plt.subplot(8, 1, i)
    sns.countplot(x=col, data=df, palette='Set2')
    plt.title(f'Count Plot of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')

plt.tight_layout()
plt.show()

from sklearn.preprocessing import LabelEncoder

categorical_columns = ['Gender', 'family_history_with_overweight', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad', 'FAVC']

encoding_mappings = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[f'{col}'] = le.fit_transform(df[col])
    encoding_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

for col, mapping in encoding_mappings.items():
    print(f"Correspondance pour {col} : {mapping}")
df

correlation_matrix = df.corr()

plt.figure(figsize=(25, 8))

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=1, linecolor='black')
plt.title('Correlation Matrix of Numerical Variables')
plt.show()

cols_to_drop = ['FAVC', 'FCVC', 'SMOKE', 'NCP', 'CH2O', 'SCC', 'TUE']
df = df.drop(columns=cols_to_drop)

df.to_csv("../Data/Data_for_trainning.csv", index=False)
colonnes = df.columns.tolist()

X = df.drop(['NObeyesdad'], axis=1)
y = df['NObeyesdad']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Create a new DataFrame with resampled data
df_resampled = pd.concat([X_resampled, y_resampled], axis=1)


# In[30]:


df_resampled.to_csv("../Data/SMOTE_for_Data.csv", index=False)


# In[32]:


# Original DataFrame: df
# Balanced DataFrame after SMOTE: df_resampled

# Summary Statistics
print("Original Data:")
print(df.describe(include='all'))
print("\nResampled Data:")
print(df_resampled.describe(include='all'))

# Histogram for Numeric Variables
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

# Bar Plot for Categorical Variables
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Distribution of Categorical Variables in Resampled Data')

sns.countplot(data=df_resampled, x='Gender', ax=axes[0])
axes[0].set_title('Gender Distribution')

sns.countplot(data=df_resampled, x='NObeyesdad', ax=axes[1])
axes[1].set_title('Label Distribution')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

X = df_resampled.drop(['NObeyesdad'], axis=1)
y = df_resampled['NObeyesdad']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'])
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

logistic_model = LogisticRegression(random_state=42, solver='lbfgs')


logistic_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_logistic = logistic_model.predict(X_test)

# Calculate probabilities for ROC-AUC score
y_prob_logistic = logistic_model.predict_proba(X_test)

# Initialize a dictionary to hold the evaluation metrics
metrics_dict = {
    'Model': ['Logistic Regression'],
    'Accuracy': [accuracy_score(y_test, y_pred_logistic)],
    'Precision': [precision_score(y_test, y_pred_logistic, average='weighted')],
    'Recall': [recall_score(y_test, y_pred_logistic, average='weighted')],
    'F1-Score': [f1_score(y_test, y_pred_logistic, average='weighted')],
    'AUC-ROC': [roc_auc_score(y_test, y_prob_logistic, multi_class='ovr', average='weighted')]
}

# Create a DataFrame to display the results
metrics_df = pd.DataFrame(metrics_dict)
metrics_df


# ## Decision Tree Model Performance

# Le modèle **Decision Tree** (arbre de décision) est une méthode d'apprentissage supervisé utilisée à la fois pour des tâches de classification et de régression. Il fonctionne en segmentant les données en sous-ensembles basés sur des règles conditionnelles, créant ainsi une structure arborescente où chaque nœud représente une décision basée sur une feature spécifique.

# In[41]:


from sklearn.tree import DecisionTreeClassifier

# Initialize the Decision Tree model
decision_tree_model = DecisionTreeClassifier(random_state=42)

# Fit the model to the training data
decision_tree_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_decision_tree = decision_tree_model.predict(X_test)

# Calculate probabilities for ROC-AUC score
y_prob_decision_tree = decision_tree_model.predict_proba(X_test)

# Update the metrics dictionary
metrics_dict['Model'].append('Decision Tree')
metrics_dict['Accuracy'].append(accuracy_score(y_test, y_pred_decision_tree))
metrics_dict['Precision'].append(precision_score(y_test, y_pred_decision_tree, average='weighted'))
metrics_dict['Recall'].append(recall_score(y_test, y_pred_decision_tree, average='weighted'))
metrics_dict['F1-Score'].append(f1_score(y_test, y_pred_decision_tree, average='weighted'))
metrics_dict['AUC-ROC'].append(roc_auc_score(y_test, y_prob_decision_tree, multi_class='ovr', average='weighted'))

# Update the DataFrame to display the results
metrics_df = pd.DataFrame(metrics_dict)

metrics_df


# ## Random Forest Model

# Le modèle **Random Forest** est un algorithme d'apprentissage supervisé puissant et polyvalent, largement utilisé pour les tâches de classification et de régression. Il repose sur le concept d’ensemble learning, où plusieurs arbres de décision (decision trees) sont construits et combinés pour améliorer la précision et réduire le risque de surapprentissage (overfitting).

# In[42]:


from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest model
random_forest_model = RandomForestClassifier(random_state=42, n_estimators=100)

# Fit the model to the training data
random_forest_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_random_forest = random_forest_model.predict(X_test)

# Calculate probabilities for ROC-AUC score
y_prob_random_forest = random_forest_model.predict_proba(X_test)

# Update the metrics dictionary
metrics_dict['Model'].append('Random Forest')
metrics_dict['Accuracy'].append(accuracy_score(y_test, y_pred_random_forest))
metrics_dict['Precision'].append(precision_score(y_test, y_pred_random_forest, average='weighted'))
metrics_dict['Recall'].append(recall_score(y_test, y_pred_random_forest, average='weighted'))
metrics_dict['F1-Score'].append(f1_score(y_test, y_pred_random_forest, average='weighted'))
metrics_dict['AUC-ROC'].append(roc_auc_score(y_test, y_prob_random_forest, multi_class='ovr', average='weighted'))

# Update the DataFrame to display the results
metrics_df = pd.DataFrame(metrics_dict)

# Sort the DataFrame by AUC-ROC in descending order
metrics_df_sorted = metrics_df.sort_values(by='AUC-ROC', ascending=False)

metrics_df_sorted


# ## Support Vector Machines (SVM)

#  
# 
# Le **Support Vector Machine** (SVM) est un algorithme d'apprentissage supervisé largement utilisé pour les tâches de classification et de régression. Il est particulièrement efficace pour les problèmes de classification binaire et peut également être adapté aux problèmes multiclasses.  
# 
# L'idée principale du SVM est de trouver l'**hyperplan optimal** qui sépare au mieux les différentes classes dans l'espace des features. Ce modèle cherche à maximiser la **marge** — la distance entre l'hyperplan et les points les plus proches de chaque classe, appelés **vecteurs de support**.  
# 
# 

# In[43]:


from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

# Initialize the SVM model
svm_model = SVC(random_state=42, probability=True)

# Fit the model to the training data
svm_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_svm = svm_model.predict(X_test)

# Calculate probabilities for ROC-AUC score
y_prob_svm = svm_model.predict_proba(X_test)

# Update the metrics dictionary
metrics_dict['Model'].append('SVM')
metrics_dict['Accuracy'].append(accuracy_score(y_test, y_pred_svm))
metrics_dict['Precision'].append(precision_score(y_test, y_pred_svm, average='weighted'))
metrics_dict['Recall'].append(recall_score(y_test, y_pred_svm, average='weighted'))
metrics_dict['F1-Score'].append(f1_score(y_test, y_pred_svm, average='weighted'))
metrics_dict['AUC-ROC'].append(roc_auc_score(y_test, y_prob_svm, multi_class='ovr', average='weighted'))

# Update the DataFrame to display the results
metrics_df = pd.DataFrame(metrics_dict)

# Sort the DataFrame by AUC-ROC in descending order
metrics_df_sorted = metrics_df.sort_values(by='AUC-ROC', ascending=False)

metrics_df_sorted


# ## k-Nearest Neighbors (k-NN)

# 
# Le **k-Nearest Neighbors** (ou k-NN) est un algorithme d'apprentissage supervisé simple mais puissant, utilisé pour les tâches de **classification** et de **régression**. Son fonctionnement repose sur une idée intuitive :  
# > **"Un point de données est classé selon la majorité des classes de ses *k* voisins les plus proches."**  
# 
# 

# In[44]:


from sklearn.neighbors import KNeighborsClassifier

# Initialize the k-NN model
knn_model = KNeighborsClassifier(n_neighbors=5)

# Fit the model to the training data
knn_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_knn = knn_model.predict(X_test)

# Calculate probabilities for ROC-AUC score
y_prob_knn = knn_model.predict_proba(X_test)

# Update the metrics dictionary
metrics_dict['Model'].append('k-NN')
metrics_dict['Accuracy'].append(accuracy_score(y_test, y_pred_knn))
metrics_dict['Precision'].append(precision_score(y_test, y_pred_knn, average='weighted'))
metrics_dict['Recall'].append(recall_score(y_test, y_pred_knn, average='weighted'))
metrics_dict['F1-Score'].append(f1_score(y_test, y_pred_knn, average='weighted'))
metrics_dict['AUC-ROC'].append(roc_auc_score(y_test, y_prob_knn, multi_class='ovr', average='weighted'))

# Update the DataFrame to display the results
metrics_df = pd.DataFrame(metrics_dict)

# Sort the DataFrame by AUC-ROC in descending order
metrics_df_sorted = metrics_df.sort_values(by='AUC-ROC', ascending=False)

metrics_df_sorted


# ## Naive Bayes

# 
# 
# Le **Naive Bayes** est un algorithme de classification probabiliste basé sur le **théorème de Bayes**. Il est particulièrement utilisé pour les tâches de classification où l'on cherche à prédire la probabilité qu'une donnée appartienne à une certaine catégorie.  
# 
# L’algorithme repose sur une hypothèse forte mais simplificatrice : **chaque feature est indépendante des autres** (d’où le terme *naive*). 
# 

# In[45]:


from sklearn.naive_bayes import GaussianNB

# Initialize the Naive Bayes model
naive_bayes_model = GaussianNB()

# Fit the model to the training data
naive_bayes_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_naive_bayes = naive_bayes_model.predict(X_test)

# Calculate probabilities for ROC-AUC score
y_prob_naive_bayes = naive_bayes_model.predict_proba(X_test)

# Update the metrics dictionary
metrics_dict['Model'].append('Naive Bayes')
metrics_dict['Accuracy'].append(accuracy_score(y_test, y_pred_naive_bayes))
metrics_dict['Precision'].append(precision_score(y_test, y_pred_naive_bayes, average='weighted'))
metrics_dict['Recall'].append(recall_score(y_test, y_pred_naive_bayes, average='weighted'))
metrics_dict['F1-Score'].append(f1_score(y_test, y_pred_naive_bayes, average='weighted'))
metrics_dict['AUC-ROC'].append(roc_auc_score(y_test, y_prob_naive_bayes, multi_class='ovr', average='weighted'))

# Update the DataFrame to display the results
metrics_df = pd.DataFrame(metrics_dict)

# Sort the DataFrame by AUC-ROC in descending order
metrics_df_sorted = metrics_df.sort_values(by='AUC-ROC', ascending=False)

metrics_df_sorted


# ## Gradient Boosting

# 
# Le **Gradient Boosting** est une technique d'apprentissage automatique supervisé, particulièrement puissante pour les tâches de classification et de régression. Il repose sur le principe de l'*ensemble learning*, c'est-à-dire qu'il construit un modèle robuste en combinant plusieurs modèles faibles (souvent des arbres de décision) pour corriger leurs erreurs successives.  
# 
# 

# In[46]:


from sklearn.ensemble import GradientBoostingClassifier

# Initialize the Gradient Boosting model
gradient_boosting_model = GradientBoostingClassifier(random_state=42)

# Fit the model to the training data
gradient_boosting_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_gradient_boosting = gradient_boosting_model.predict(X_test)

# Calculate probabilities for ROC-AUC score
y_prob_gradient_boosting = gradient_boosting_model.predict_proba(X_test)

# Update the metrics dictionary
metrics_dict['Model'].append('Gradient Boosting')
metrics_dict['Accuracy'].append(accuracy_score(y_test, y_pred_gradient_boosting))
metrics_dict['Precision'].append(precision_score(y_test, y_pred_gradient_boosting, average='weighted'))
metrics_dict['Recall'].append(recall_score(y_test, y_pred_gradient_boosting, average='weighted'))
metrics_dict['F1-Score'].append(f1_score(y_test, y_pred_gradient_boosting, average='weighted'))
metrics_dict['AUC-ROC'].append(roc_auc_score(y_test, y_prob_gradient_boosting, multi_class='ovr', average='weighted'))

# Update the DataFrame to display the results
metrics_df = pd.DataFrame(metrics_dict)

# Sort the DataFrame by AUC-ROC in descending order
metrics_df_sorted = metrics_df.sort_values(by='AUC-ROC', ascending=False)

metrics_df_sorted


# ## XGBoost Classifier

# 
# 
# Le **XGBoost Classifier** (*Extreme Gradient Boosting*) est un algorithme de machine learning basé sur les arbres de décision, spécialement conçu pour l'apprentissage supervisé et très apprécié pour sa rapidité et ses performances élevées.  
# 
# Il repose sur le principe du **boosting**, une technique d'ensemble qui construit plusieurs arbres de décision de manière séquentielle, où chaque nouvel arbre corrige les erreurs des arbres précédents. Ce processus permet de réduire les biais et la variance, aboutissant à un modèle puissant et précis.  
# 
# Les points forts du **XGBoost Classifier** incluent :  
# - **Vitesse et efficacité** : Grâce à son optimisation du calcul parallèle et à sa gestion intelligente de la mémoire.  
# - **Régularisation** : Intègre les termes de L1 et L2 pour éviter le surapprentissage (*overfitting*).  
# - **Flexibilité** : Compatible avec les fonctions de coût personnalisées et plusieurs options pour gérer les valeurs manquantes.  
# - **Importance des features** : Fournit des mesures claires pour identifier les variables les plus influentes dans les prédictions.  
# 
# 
# 

# In[47]:


from xgboost import XGBClassifier

# Initialisation du modèle XGBoost
xgboost_model = XGBClassifier(eval_metric='mlogloss')

# Entraînement du modèle
xgboost_model.fit(X_train, y_train)

# Prédictions sur les données de test
y_pred_xgboost = xgboost_model.predict(X_test)

# Probabilités pour le calcul de l'AUC-ROC
y_prob_xgboost = xgboost_model.predict_proba(X_test)

# Mise à jour des métriques pour XGBoost
metrics_dict['Model'].append('XGBoost')
metrics_dict['Accuracy'].append(accuracy_score(y_test, y_pred_xgboost))
metrics_dict['Precision'].append(precision_score(y_test, y_pred_xgboost, average='weighted'))
metrics_dict['Recall'].append(recall_score(y_test, y_pred_xgboost, average='weighted'))
metrics_dict['F1-Score'].append(f1_score(y_test, y_pred_xgboost, average='weighted'))
metrics_dict['AUC-ROC'].append(roc_auc_score(y_test, y_prob_xgboost, multi_class='ovr', average='weighted'))

# Affichage des résultats
metrics_df = pd.DataFrame(metrics_dict)
metrics_df_sorted = metrics_df.sort_values(by='AUC-ROC', ascending=False)
metrics_df_sorted


# ## LightGBM Classifier

# 
# 
# Le **LightGBM Classifier** (Light Gradient Boosting Machine) est un algorithme de machine learning puissant et optimisé pour les tâches de classification supervisée. Développé par Microsoft, LightGBM repose sur le principe du *gradient boosting* mais se distingue par plusieurs caractéristiques clés qui le rendent particulièrement performant :  
# 
# - **Efficacité et rapidité** : LightGBM utilise une technique appelée *leaf-wise growth* (croissance basée sur les feuilles), ce qui lui permet de converger plus rapidement et de gérer efficacement de grands ensembles de données.  
# - **Gestion des grandes dimensions** : Il prend en charge les données avec un grand nombre de features tout en optimisant l’utilisation de la mémoire.  
# - **Prise en charge des données catégoriques** : Contrairement à certains modèles, LightGBM peut gérer directement les variables catégoriques sans avoir besoin de techniques de transformation comme le *one-hot encoding*.  
# - **Réduction du surapprentissage** : Grâce à des paramètres comme le *max depth* et le *min data in leaf*, il offre une régularisation efficace pour éviter le surapprentissage.  
# 
# 

# In[48]:


from lightgbm import LGBMClassifier

# Initialisation du modèle LightGBM
lightgbm_model = LGBMClassifier(
    n_estimators=100,      # Réduire le nombre d'itérations
    max_depth=7,           # Limiter la profondeur des arbres
    min_data_in_leaf=20,   # Augmenter le nombre de données minimum par feuille
    min_split_gain=0.01,   # Augmenter le gain minimum pour effectuer un split
    class_weight='balanced' # Si les classes sont déséquilibrées
)


# Entraînement du modèle
lightgbm_model.fit(X_train, y_train)

# Prédictions sur les données de test
y_pred_lightgbm = lightgbm_model.predict(X_test)

# Probabilités pour le calcul de l'AUC-ROC
y_prob_lightgbm = lightgbm_model.predict_proba(X_test)

# Mise à jour des métriques pour LightGBM
metrics_dict['Model'].append('LightGBM')
metrics_dict['Accuracy'].append(accuracy_score(y_test, y_pred_lightgbm))
metrics_dict['Precision'].append(precision_score(y_test, y_pred_lightgbm, average='weighted'))
metrics_dict['Recall'].append(recall_score(y_test, y_pred_lightgbm, average='weighted'))
metrics_dict['F1-Score'].append(f1_score(y_test, y_pred_lightgbm, average='weighted'))
metrics_dict['AUC-ROC'].append(roc_auc_score(y_test, y_prob_lightgbm, multi_class='ovr', average='weighted'))

# Affichage des résultats
metrics_df = pd.DataFrame(metrics_dict)
metrics_df_sorted = metrics_df.sort_values(by='AUC-ROC', ascending=False)
metrics_df_sorted


# ## CatBoost Classifier

# 
# 
# Le **CatBoost Classifier** est un algorithme de *Gradient Boosting* développé par Yandex, spécialement conçu pour gérer efficacement les variables catégorielles tout en offrant des performances remarquables.  
# 
# Ce modèle fait partie des algorithmes de *boosting* qui fonctionnent en construisant une série d'arbres de décision successifs, chaque nouvel arbre tentant de corriger les erreurs de prédiction des arbres précédents. Ce processus itératif permet au modèle d'atteindre une grande précision, souvent supérieure à celle des modèles classiques.
# 
# 

# In[49]:


from catboost import CatBoostClassifier

# Initialisation du modèle CatBoost
catboost_model = CatBoostClassifier(silent=True)

# Entraînement du modèle
catboost_model.fit(X_train, y_train)

# Prédictions sur les données de test
y_pred_catboost = catboost_model.predict(X_test)

# Probabilités pour le calcul de l'AUC-ROC
y_prob_catboost = catboost_model.predict_proba(X_test)

# Mise à jour des métriques pour CatBoost
metrics_dict['Model'].append('CatBoost')
metrics_dict['Accuracy'].append(accuracy_score(y_test, y_pred_catboost))
metrics_dict['Precision'].append(precision_score(y_test, y_pred_catboost, average='weighted'))
metrics_dict['Recall'].append(recall_score(y_test, y_pred_catboost, average='weighted'))
metrics_dict['F1-Score'].append(f1_score(y_test, y_pred_catboost, average='weighted'))
metrics_dict['AUC-ROC'].append(roc_auc_score(y_test, y_prob_catboost, multi_class='ovr', average='weighted'))

# Affichage des résultats
metrics_df = pd.DataFrame(metrics_dict)
metrics_df_sorted = metrics_df.sort_values(by='AUC-ROC', ascending=False)
metrics_df_sorted


# ## Analyse des résultats :
# 
# 1. **Modèles les plus performants** :  
#    - **Random Forest** a la meilleure précision globale (**Accuracy** = 99.04%), avec des scores presque parfaits pour toutes les autres métriques.
#    - **CatBoost** suit de très près, avec une **Accuracy** de 98.91% et le meilleur score **AUC-ROC** (0.999924), ce qui montre qu'il distingue très bien les classes.
# 
# 2. **Modèles Boostés** :
#    - **LightGBM** et **XGBoost** affichent aussi d'excellents résultats (près de 99% pour toutes les métriques), prouvant la puissance des algorithmes basés sur les arbres décisionnels boostés.
#    
# 3. **Modèles classiques** :
#    - **SVM** (Support Vector Machine) a des scores légèrement plus bas (**Accuracy** = 92.92%), ce qui suggère qu'il pourrait avoir du mal avec des données complexes ou mal séparables.
#    - **Naive Bayes** et **Logistic Regression** obtiennent des résultats corrects (entre 91% et 92%) mais restent en retrait par rapport aux modèles boostés.
#    
# 4. **Modèles les moins performants** :
#    - **k-NN** est clairement le modèle le moins performant ici, avec une **Accuracy** de 85.3% et un **AUC-ROC** de 0.97. Cela peut indiquer qu'il est sensible au bruit ou que les données ne sont pas optimales pour ce type de modèle.
#    
# 5. **Équilibre précision/recall** :
#    - Tous les meilleurs modèles maintiennent un équilibre entre **Precision** et **Recall**, ce qui est essentiel pour minimiser à la fois les faux positifs et les faux négatifs.  
#    
# 6. **AUC-ROC** :
#    - Cette métrique étant proche de 1 pour tous les modèles boostés, cela indique qu'ils séparent efficacement les classes et qu’ils sont particulièrement bien adaptés à cette tâche.
# 
# ## Conclusion :  
# Les modèles boostés (Random Forest, CatBoost, XGBoost, LightGBM) dominent le classement, suggérant qu'ils gèrent très bien la complexité des données et minimisent les erreurs. Pour un compromis entre rapidité et précision, **Random Forest** et **CatBoost** semblent les choix les plus robustes.
# 
# Pour la suite nous avons choisi le **CatBoost** 

# ## Optimization
# 

# L'optimisation de la mémoire est essentielle pour gérer efficacement de grands volumes de données. Dans cette section, nous avons mis en place une stratégie de réduction de la taille des colonnes numériques et catégorielles. Cela permet non seulement de minimiser l'empreinte mémoire, mais aussi d'accélérer les calculs sans compromettre la précision des résultats.

# In[51]:


def optimize_memory(df):
    # Afficher la mémoire utilisée avant l'optimisation
    print(f"Memory usage before optimization: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Pour chaque colonne du DataFrame, ajuster le type de données pour économiser de la mémoire
    for column in df.columns:
        col_type = df[column].dtype
        
        # Optimiser les colonnes numériques
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




