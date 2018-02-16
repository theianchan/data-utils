import pandas as pd
from IPython.display import display


def display_all(df):
    """Display all rows and columns in a DataFrame"""
    with pd.option_context("display.max_rows", 1000):
        with pd.option_context("display.max_columns", 1000):
            display(df)


def show_column_nulls(df):
    """Show null values frequencies in a DataFrame
    by column"""

    df_null = df.isnull().sum() / len(df)
    return pd.DataFrame(df_null, columns=["Null Frequency"])


def describe_qual(df):
    """Describes qualitative data in a DataFrame"""

    categorical = df.dtypes[df.dtypes == "object"].index
    df[categorical].describe()


def describe_quant(df):
    """Describes quantitative data in a DataFrame
    (Uses default behavior of df.describe but
    added for notational consistency)"""

    df.describe()


def value_frequencies(df, column):
    """Returns the counts and frequencies
    of unique values in a DataFrame"""

    df_val = pd.DataFrame(df[column].value_counts())
    df_val = df_val.rename(columns={column: "Count"})
    df_val["Frequency"] = df_val["Count"] / len(df) * 100
    return df_val


def show_qual(df, show_head=5, show_tail=3):
    """Prints the qualitative values in a DataFrame
    with the highest and lowest unique value counts"""

    categorical = df.dtypes[df.dtypes == "object"].index
    for column in categorical:
        df_val = value_frequencies(df, column)
        if (df[column].nunique() >= 10):
            df_val = pd.concat([
                df_val.head(show_head),
                pd.DataFrame({
                    "Count": "...",
                    "Frequency": "..."
                }, index=["..."]),
                df_val.tail(show_tail)
            ])
        print("{}\n{}\n".format(column, df_val))
