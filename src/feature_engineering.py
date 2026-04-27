import pandas as pd

def create_lag_features(df):
    df['Inflation_Lag1'] = df.groupby('Country')['Inflation_CPI'].shift(1)
    df['Inflation_Lag2'] = df.groupby('Country')['Inflation_CPI'].shift(2)
    df['GDP_Growth_Lag1'] = df.groupby('Country')['GDP_Growth'].shift(1)

    df = df.dropna()

    return df

def encode_country(df):
    df = pd.get_dummies(df, columns=['Country'])
    return df

def prepare_features(df):
    df = create_lag_features(df)
    df = encode_country(df)
    return df