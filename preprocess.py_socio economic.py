import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path="data/socio_economic.csv"):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    df = df.dropna()
    features = df.drop("crisis", axis=1)
    labels = df["crisis"]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled, labels, scaler
