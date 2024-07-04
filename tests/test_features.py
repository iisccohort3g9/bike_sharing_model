
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from bike_sharing_model.config.core import config
# from bike_sharing_model.processing.features import age_col_tfr


# def test_age_variable_transformer(sample_input_data):
#     # Given
#     transformer = age_col_tfr(
#         variables=config.model_config.age_var,  # cabin
#     )
#     assert np.isnan(sample_input_data[0].loc[709,'Age'])

#     # When
#     subject = transformer.fit(sample_input_data[0]).transform(sample_input_data[0])

#     # Then
#     assert subject.loc[709,'Age'] == 21

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