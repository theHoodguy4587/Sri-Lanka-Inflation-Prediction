import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)


def select_columns(df):
    return df[[ 'country_name',
                'year',
                'Inflation (CPI %)',
                'GDP Growth (% Annual)',
                'Inflation (GDP Deflator, %)',
                'GDP per Capita (Current USD)']]

#Rename columns
def rename_columns(df):
    df.columns = ['Country',
              'Year',
              'Inflation_CPI',
              'GDP_Growth',
              'Inflation_GDP_Deflator',
              'GDP_per_Capita']

def clean_data(df):


    #drop missing target

    df = df.dropna(subset=['Inflation_CPI']).copy()

    #fill missing values for economic indicators
    cols =['GDP_Growth',
           'Inflation_GDP_Deflator',
           'GDP_per_Capita']

    df.loc[:, cols] = df.groupby('Country')[cols].ffill().bfill()

    #drop remaning missing values
    df = df.dropna()

    return df