import pandas as pd

def load_data(file_path):
    try:
        file_path = '../Data/ObesityDataSet_raw_and_data_sinthetic.csv'
        df = pd.read_csv(file_path)
        print("Data successfully loaded.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

