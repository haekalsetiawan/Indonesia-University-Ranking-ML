import pandas as pd

def load_and_clean_data(filepath):
    # Baca file CSV
    df = pd.read_csv(filepath)
    return df

def add_features(df):
    # Contoh penambahan fitur
    df['Region'] = df['Town'].apply(lambda x: 'Region 1' if x in ['Town1', 'Town2'] else 'Region 2')
    return df
