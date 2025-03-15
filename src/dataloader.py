import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = fetch_ucirepo(id=544)

# Data (as pandas dataframes)
X = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.features
y = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.targets

file_path = '../Data/ObesityDataSet_raw_and_data_sinthetic.csv'
df = pd.read_csv(file_path)