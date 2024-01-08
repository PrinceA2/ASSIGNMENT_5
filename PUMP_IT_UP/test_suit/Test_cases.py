# Importing important libraries

import pandas as pd
import numpy as np
from src.preprocessing.data_cleaning import drop_columns, drop_null_values, label_encode_column, creating_location_feature, calculate_age_in_years, replace_zeros_with_mean, count_zeros_in_column
import pytest

# Decorator
@pytest.fixture


def sample_dataframe():
    data = {
        'A': [1, 2, 3, 0, 5],
        'B': [0, 2, 3, 4, 5],
        'C': ['X', 'Y', 'X', 'Y', 'X']
    }
    return pd.DataFrame(data)

# Test_case function for dropping the columns
def test_drop_columns(sample_dataframe):
    columns_to_drop = ['A', 'B']
    result_df = drop_columns(sample_dataframe, columns_to_drop)
    expected_df = pd.DataFrame({'C': ['X', 'Y', 'X', 'Y', 'X']})
    pd.testing.assert_frame_equal(result_df, expected_df)

# Test_case function for dropping null values
def test_drop_null_values(sample_dataframe):
    column_name = 'A'
    result_df = drop_null_values(sample_dataframe, column_name)
    expected_df = sample_dataframe[sample_dataframe[column_name].notnull()]
    pd.testing.assert_frame_equal(result_df, expected_df)

# Test_case function for counting zeros in a column
def test_count_zeros_in_column(sample_dataframe):
    column_name = 'A'
    result_count = count_zeros_in_column(sample_dataframe, column_name)
    expected_count = (sample_dataframe[column_name] == 0).sum()
    assert result_count == expected_count
