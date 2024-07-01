"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from bike_sharing_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    expected_no_predictions = 3476 #X_test.shape

    # When
    result = make_prediction(input_data=sample_input_data[0])

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, np.ndarray)
    assert isinstance(predictions[0], np.int64)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    _predictions = list(predictions)
    y_true = sample_input_data[1]
    accuracy = accuracy_score(_predictions, y_true)
    assert accuracy > 0.8
    
def test_make_prediction_empty_input():
    # Given
    sample_input_data = pd.DataFrame()

    # When
    result = make_prediction(input_data=sample_input_data)

    # Then
    assert result.get("predictions") is None
    assert result.get("errors") is not None
    
def test_make_prediction_incorrect_data_types():
    # Given
    sample_input_data = {"holiday": ["Yes", "No", "Yes"], "hr": ["6am", "4am", "11am"], "weathersit": ["Mist", "Clear", "Mist"]}

    # When
    result = make_prediction(input_data=sample_input_data)

    # Then
    assert result.get("predictions") is None
    assert result.get("errors") is not None

