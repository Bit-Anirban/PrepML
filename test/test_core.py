import pytest
import pandas as pd
import numpy as np
from engine.core2 import DataCleanerEngine

# ==========================================
# FIXTURES
# ==========================================
@pytest.fixture
def sample_data():
    """Creates a mock dataset simulating messy corporate carbon footprint data."""
    return pd.DataFrame({
        "company_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "carbon_emissions_tons": [150.5, np.nan, 300.0, 100.2, 50.0, 400.1, 250.0, np.nan, 80.5, 90.0, 110.0],
        "revenue_str": ["$43.00", "-$15.50", "1,500%", "$100", "200.5", "-300", "$50", "$60", "-$10", "$80", "$90"],
        "is_compliant": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes"]
    })

@pytest.fixture
def engine(sample_data):
    """Initializes the DataCleanerEngine with the mock dataset."""
    return DataCleanerEngine(sample_data, "mock_data.csv")


# ==========================================
# TEST CASES
# ==========================================
def test_engine_initialization(engine, sample_data):
    """Ensure the engine loads data correctly and copies to clean_df."""
    assert engine.file_name == "mock_data.csv"
    assert engine.raw_df.shape == sample_data.shape
    assert engine.clean_df.shape == sample_data.shape
    assert len(engine.audit_log) > 0

def test_drop_columns(engine):
    """Ensure columns are permanently dropped from clean_df."""
    assert "company_id" in engine.clean_df.columns
    engine.drop_columns(["company_id"])
    assert "company_id" not in engine.clean_df.columns
    assert "df.drop(columns=['company_id'], inplace=True)" in engine.audit_log[-1]

def test_handle_missing_mean(engine):
    """Ensure missing values are correctly filled with the column mean."""
    col = "carbon_emissions_tons"
    assert engine.clean_df[col].isna().sum() == 2
    
    # The mean of the 9 non-null values is approx 170.14
    engine.handle_missing(col, "Fill with Mean")
    
    assert engine.clean_df[col].isna().sum() == 0
    assert "fillna" in engine.audit_log[-1]

def test_clean_string_to_numeric(engine):
    """Ensure the regex successfully strips currency/percentages but keeps negatives."""
    col = "revenue_str"
    engine.clean_string_to_numeric(col)
    
    # Check if the column was converted to float
    assert pd.api.types.is_numeric_dtype(engine.clean_df[col])
    
    # Check specific values to ensure negatives and decimals were preserved
    values = engine.clean_df[col].tolist()
    assert values[0] == 43.00   # $43.00 -> 43.0
    assert values[1] == -15.50  # -$15.50 -> -15.5
    assert values[2] == 1500.0  # 1,500% -> 1500.0

def test_detect_target_type(engine):
    """Ensure the engine correctly identifies binary vs regression targets."""
    # is_compliant has 2 unique values ("Yes", "No")
    cat_target = engine.detect_target_type("is_compliant")
    assert cat_target["type"] == "Categorical"
    assert cat_target["subtype"] == "Binary"
    
    # company_id is numerical with > 10 unique values
    num_target = engine.detect_target_type("company_id")
    assert num_target["type"] == "Numerical"
    assert num_target["subtype"] == "Regression"

def test_audit_log_generation(engine):
    """Ensure the code export string builds properly."""
    engine.drop_columns(["company_id"])
    generated_code = engine.get_generated_code()
    
    assert "import pandas as pd" in generated_code
    assert "pd.read_csv('mock_data.csv')" in generated_code
    assert "df.drop(columns=['company_id']" in generated_code