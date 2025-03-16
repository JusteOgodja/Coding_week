from data_preprocessing import preprocess_data

file_path = './Data/ObesityDataSet_raw_and_data_sinthetic.csv'
"
X_train, X_test, y_train, y_test = preprocess_data(file_path)

print("Taille des ensembles de données :")
print("Train:", X_train.shape, "Test:", X_test.shape)


