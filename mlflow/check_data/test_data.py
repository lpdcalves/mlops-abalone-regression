import pandas as pd
import scipy.stats

        
# Determinstic Test
def test_column_presence_and_type(data):
    # Disregard the reference dataset
    _, df = data

    required_columns = {
        "Sex_M": pd.api.types.is_int64_dtype,
        "Sex_I": pd.api.types.is_int64_dtype,
        "Sex_F": pd.api.types.is_int64_dtype,
        "Length": pd.api.types.is_float_dtype,
        "Height": pd.api.types.is_float_dtype,
        "Whole weight": pd.api.types.is_float_dtype,
        "Age": pd.api.types.is_float_dtype,
    }

    # Check column presence
    assert set(df.columns.values).issuperset(set(required_columns.keys()))

    for col_name, format_verification_funct in required_columns.items():
        assert format_verification_funct(df[col_name]), f"Column {col_name} failed test {format_verification_funct}"