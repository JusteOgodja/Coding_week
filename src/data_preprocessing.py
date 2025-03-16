import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib


def load_data(file_path):
    """Charge le dataset depuis un fichier CSV."""
    df = pd.read_csv(file_path)
    return df


def explore_data(df):
    """Affiche les informations de base sur les données."""
    print("Shape of data:", df.shape)
    print("Data types:\n", df.dtypes)
    print("Missing values:\n", df.isnull().sum())
    print("Summary statistics:\n", df.describe(include="all"))


def visualize_distributions(df, numeric_columns):
    """Affiche les distributions des variables numériques."""
    sns.set_style("whitegrid")
    plt.figure(figsize=(20, 10))

    for i, col in enumerate(numeric_columns, start=1):
        plt.subplot(2, 4, i)
        sns.histplot(df[col], bins=20, kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def detect_outliers(df, column):
    """Détecte les valeurs aberrantes en utilisant l'IQR."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]


def remove_outliers(df, age_limit=50):
    """Supprime les valeurs aberrantes et filtre certaines données."""
    df = df[df["Age"] <= age_limit]
    return df


def encode_categorical_features(df, categorical_columns):
    """Encode les variables catégoriques avec LabelEncoder."""
    encoding_mappings = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoding_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    return df, encoding_mappings


def correlation_heatmap(df):
    """Affiche une heatmap des corrélations."""
    correlation_matrix = df.corr()
    plt.figure(figsize=(25, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=1, linecolor='black')
    plt.title('Correlation Matrix')
    plt.show()


def apply_smote(X, y):
    """Applique la technique SMOTE pour équilibrer les classes."""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def scale_features(X):
    """Normalise les caractéristiques en utilisant StandardScaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.pkl")  # Sauvegarde du scaler
    return X_scaled


def split_data(X, y, test_size=0.3):
    """Sépare les données en ensembles d'entraînement et de test."""
    return train_test_split(X, y, test_size=test_size, random_state=42)


def preprocess_data(file_path):
    """Pipeline complet de prétraitement des données."""
    df = load_data(file_path)

    explore_data(df)

    numeric_columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    visualize_distributions(df, numeric_columns)

    df = remove_outliers(df)

    categorical_columns = ['Gender', 'family_history_with_overweight', 'CAEC', 'SMOKE',
                           'SCC', 'CALC', 'MTRANS', 'NObeyesdad', 'FAVC']
    df, encoding_mappings = encode_categorical_features(df, categorical_columns)

    correlation_heatmap(df)

    # Suppression des colonnes non pertinentes
    cols_to_drop = ['FAVC', 'FCVC', 'SMOKE', 'NCP', 'CH2O', 'SCC', 'TUE']
    df = df.drop(columns=cols_to_drop)

    df.to_csv("../Data/Data_for_trainning.csv", index=False)

    X = df.drop(['NObeyesdad'], axis=1)
    y = df['NObeyesdad']

    X_resampled, y_resampled = apply_smote(X, y)

    df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
    df_resampled.to_csv("../Data/SMOTE_for_Data.csv", index=False)

    X_scaled = scale_features(X_resampled)

    return split_data(X_scaled, y_resampled)


if __name__ == "__main__":
    file_path = "./Data/ObesityDataSet_raw_and_data_sinthetic.csv"
    X_train, X_test, y_train, y_test = preprocess_data(file_path)
    print("Preprocessing completed!")


