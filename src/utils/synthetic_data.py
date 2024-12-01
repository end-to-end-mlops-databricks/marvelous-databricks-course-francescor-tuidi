"""Module to generate synthetic data based on the distribution of the original data."""

import numpy as np
import pandas as pd


def create_synthetic_data(df, num_rows=100, id_column="Rank"):
    """Create synthetic data based on the distribution of the original data.

    Args:
        df (pd.DataFrame): Original data.
        num_rows (int, optional): Number of rows to generate. Defaults to 100.
        id_column (str, optional): Name of the ID column. Defaults to "Rank".

    Returns:
        pd.DataFrame: Synthetic data.
    """
    existing_ids = set(int(id) for id in df[id_column])
    synthetic_data = pd.DataFrame()

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]) and column != id_column:
            if column in ["YearBuilt", "YearRemodAdd"]:
                synthetic_data[column] = np.random.randint(
                    df[column].min(), df[column].max() + 1, num_rows
                )  # Years between existing values
            else:
                mean, std = df[column].mean(), df[column].std()
                synthetic_data[column] = np.random.normal(mean, std, num_rows)

        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            synthetic_data[column] = np.random.choice(
                df[column].unique(), num_rows, p=df[column].value_counts(normalize=True)
            )

        elif isinstance(df[column].dtype, pd.CategoricalDtype) or isinstance(df[column].dtype, pd.StringDtype):
            synthetic_data[column] = np.random.choice(
                df[column].unique(), num_rows, p=df[column].value_counts(normalize=True)
            )
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            min_date, max_date = df[column].min(), df[column].max()
            if min_date < max_date:
                synthetic_data[column] = pd.to_datetime(np.random.randint(min_date.value, max_date.value, num_rows))
            else:
                synthetic_data[column] = [min_date] * num_rows

        else:
            synthetic_data[column] = np.random.choice(df[column], num_rows)

    new_ids = []
    i = max(existing_ids) + 1 if existing_ids else 1
    while len(new_ids) < num_rows:
        if i not in existing_ids:
            new_ids.append(str(i))  # Convert numeric ID to string
        i += 1
    synthetic_data[id_column] = new_ids

    return synthetic_data
