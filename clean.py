import numpy as np
import pandas as pd


def convert_to_categories(df, column, categories):
    """Returns a DataFrame with the specified column
    converted to categorical values"""

    categorical = pd.Categorical(df[column], ordered=True)
    categorical = categorical.rename_categories(categories)
    df[column] = categorical
    return df


def impute_with_median(df, column):
    """Returns a DataFrame with missing values in
    the specified column replaced with the column
    median"""

    col_mean = df[column].median()
    imputed = np.where(df[column].isnull(), col_mean, df[column])
    df[column] = imputed
    return df
