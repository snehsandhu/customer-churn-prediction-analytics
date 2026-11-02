"""Small, valid Telco-like records used exclusively for offline unit tests."""
import pandas as pd
import pytest

from churn_analytics.cleaning import INTERNET_ADDONS, clean_dataset
from churn_analytics.data import EXPECTED_COLUMNS


@pytest.fixture
def raw_customers():
    """Eight records covering boundaries, missing charges and service categories."""
    frame = pd.DataFrame({
        "customerID": [f"C{i}" for i in range(8)],
        "gender": ["Female", "Male"] * 4,
        "SeniorCitizen": [0, 1] * 4,
        "Partner": ["No", "Yes"] * 4,
        "Dependents": ["No"] * 8,
        "tenure": [0, 6, 12, 13, 24, 36, 60, 72],
        "PhoneService": ["Yes", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes"],
        "MultipleLines": ["No", "Yes", "No phone service", "No", "Yes", "No", "Yes", "No"],
        "InternetService": ["DSL", "Fiber optic", "No", "DSL", "Fiber optic", "No", "Fiber optic", "DSL"],
        "Contract": ["Month-to-month", "Month-to-month", "One year", "One year", "Two year", "Two year", "Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes"] * 8,
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"] * 2,
        "MonthlyCharges": [20, 49, 50, 79.99, 80, 95, 110, 30],
        "TotalCharges": [" ", "294", "600", "1039.87", "1920", "3420", "6600", "2160"],
        "Churn": ["No", "Yes", "Yes", "No", "Yes", "No", "No", "No"],
    })
    for column in INTERNET_ADDONS:
        frame[column] = ["No", "No", "No internet service", "Yes", "Yes", "No internet service", "Yes", "No"]
    return frame.loc[:, EXPECTED_COLUMNS]


@pytest.fixture
def cleaned_customers(raw_customers):
    """Analytical form of the small fixture, never used as portfolio evidence."""
    return clean_dataset(raw_customers)
