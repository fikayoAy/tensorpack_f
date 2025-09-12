import pandas as pd
import numpy as np
import pytest

from tensorpack import _find_semantic_connections_between_datasets, _GENERAL_METADATA


def test_tabular_semantic_connections_basic():
    # Create two small DataFrames with overlapping column names and values
    df1 = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['alice', 'bob', 'carol', 'dave', 'eve'],
        'value': [10, 20, 30, 40, 50]
    })

    df2 = pd.DataFrame({
        'id': [3, 4, 5, 6, 7],
        'fullname': ['carol', 'dave', 'eve', 'frank', 'grace'],
        'amount': [300, 400, 500, 600, 700]
    })

    # Register DataFrames in the general metadata so the helper can find them
    _GENERAL_METADATA[id(df1)] = {'dataframe': df1, 'columns': list(df1.columns), 'file_path': 'a.csv'}
    _GENERAL_METADATA[id(df2)] = {'dataframe': df2, 'columns': list(df2.columns), 'file_path': 'b.csv'}

    source_info = {'data_type': 'tabular', 'file_info': {'file_path': 'a.csv', 'columns': list(df1.columns), 'file_name': 'a.csv'}}
    target_info = {'data_type': 'tabular', 'file_info': {'file_path': 'b.csv', 'columns': list(df2.columns), 'file_name': 'b.csv'}}

    connections = _find_semantic_connections_between_datasets(df1, df2, source_info, target_info)

    # Expect at least one connection (table summary or column match)
    assert isinstance(connections, list)
    assert len(connections) > 0, "Expected at least one semantic connection for overlapping tabular data"

    # Clean up metadata
    del _GENERAL_METADATA[id(df1)]
    del _GENERAL_METADATA[id(df2)]
