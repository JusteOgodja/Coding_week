# **Application d'Aide à la Décision Médicale – Estimation du Risque d’Obésité**  

## **Objectif du Projet**  
Ce projet vise à développer un outil d’aide à la décision médicale permettant d’évaluer le risque d’obésité chez les patients à l’aide de modèles de machine learning explicables (SHAP). L’objectif est de fournir aux professionnels de santé un système précis, interprétable et efficace.  



## **Prétraitement des Données**  

### **Gestion des valeurs manquantes**  
L’analyse du dataset n’a révélé aucune valeur manquante critique, évitant ainsi la nécessité d’une imputation.  

### **Gestion des outliers**  
Les valeurs aberrantes ont été détectées à l’aide de l’IQR et visualisées via des boxplots. Les outliers extrêmes ont été supprimés pour la variable **Âge**, notamment ceux supérieurs à 50 ans.  

### **Analyse de la corrélation**  
Les variables fortement corrélées ont été identifiées, tandis que celles dont la corrélation était proche de 0 ont été supprimées, car elles avaient un impact négligeable sur la performance des modèles et risquaient d’introduire du bruit. Les variables éliminées sont : **FAVC, FCVC, SMOKE, NCP, CH2O, SCC, TUE**.  

### **Normalisation des entrées**  
Afin d’homogénéiser l’échelle des variables, nous avons appliqué **StandardScaler**.  



## **Feature Engineering**  

### **Création de nouvelles variables**  
Pour améliorer les performances des modèles, nous avons ajouté :  
- **BMI (Indice de Masse Corporelle)** : *BMI = Poids / (Taille²)*  

### **Encodage des variables catégoriques**  
Les variables catégoriques ont été transformées via **Label Encoding**.  



## **Équilibrage du Dataset et Gestion du Déséquilibre des Classes**  

### **Le dataset était-il équilibré ?**  
Le dataset comprenait **7 classes** correspondant à différents niveaux d’obésité, avec une répartition relativement homogène (**≈12%-16%** par classe). De légers déséquilibres ont cependant été constatés.  

### **Méthode de correction**  
Pour atténuer ce déséquilibre, nous avons appliqué **SMOTE (Synthetic Minority Over-sampling Technique)** afin de générer des exemples supplémentaires pour les classes sous-représentées.  

### **Impact de la correction**  
- Amélioration des performances du modèle, notamment sur le **recall** des classes minoritaires.  
- Réduction du biais en faveur des classes majoritaires, garantissant des prédictions plus équitables.  

 

## **Meilleur Modèle et Performances**  

### **Modèles testés**  
Nous avons évalué plusieurs algorithmes :  
- **Random Forest Classifier**  
- **CatBoost Classifier**  
- **LightGBM Classifier**  
- **Decision Tree**  
- **Logistic Regression**  
- **Support Vector Machines (SVM)**  
- **k-Nearest Neighbors (k-NN)**  
- **Gradient Boosting**  
- **Naive Bayes**  

### **Meilleur modèle**  
Le **CatBoost Classifier** s’est démarqué par sa capacité à capturer des interactions complexes entre les variables.  

### **Performances du modèle final**  

| Métrique       | Score     |  
|---------------|----------|  
| **Accuracy**  | 98.91%   |  
| **Precision** | 98.92%   |  
| **Recall**    | 98.91%   |  
| **F1-Score**  | 98.91%   |  
| **ROC-AUC**   | 99.99%   |  



## **Variables Médicales les Plus Influentes (SHAP)**  

### **Facteurs les plus influents**  
L’analyse SHAP a permis d’identifier les variables ayant le plus d’impact sur la classification de l’obésité :  

1. **Weight (Poids)**  
2. **Age (Âge)**  
3. **BMI (Indice de Masse Corporelle)**  
4. **Height (Taille)**  
5. **CALC (Fréquence de consommation d’alcool)**  

### **Explications SHAP**  
- **Le poids** est le principal facteur influençant la classification de l’obésité.  
- **L’âge avancé** joue un rôle dans certains cas d’obésité liée à des facteurs métaboliques.  
- **Un BMI élevé** est fortement associé au risque d’obésité, bien qu’il soit en partie redondant avec le poids.  
- **La taille** peut moduler l’impact du poids dans la classification.  
- **Une consommation fréquente d’alcool** est corrélée à un risque accru d’obésité.  



## **Insights du Prompt Engineering**  

### **Comment le prompt engineering a-t-il été utilisé ?**  
Nous avons exploité le **prompt engineering** pour optimiser l’analyse des données et la génération de code.  

#### **Exemples d’optimisation :**  
1. **Optimisation de la gestion de la mémoire** → Génération d’une fonction d’optimisation automatique des types de données.  
2. **Détection des outliers** → Génération d’un script efficace basé sur l’IQR et le Boxplot via un prompt dédié.  
3. **Interprétation avec SHAP** → Création de visualisations claires et explicites des résultats SHAP grâce à des prompts bien conçus.  


