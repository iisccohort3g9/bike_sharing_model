import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd

_version = "0.0.1"
from bike_sharing_model.config import config
from bike_sharing_model.processing import load_pipeline
from bike_sharing_model.processing import pre_pipeline_preparation

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
bike_sharing_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    #validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    #validated_data=validated_data.reindex(columns=['Pclass','Sex','Age','Fare', 'Embarked','FamilySize','Has_cabin','Title'])
    preprocessed_data = pre_pipeline_preparation(data_frame=pd.DataFrame(input_data))
    print("$$$$$$$$$$$$$$$$$$$$$$$$")
    print(preprocessed_data.columns)
    print("$$$$$$$$$$$$$$$$$$$$$$$$")
    validated_data=preprocessed_data.reindex(columns=config.model_config.features)
    #print(validated_data)
    results = {"predictions": None, "version": _version}
    
    predictions = bike_sharing_pipe.predict(validated_data)

    results = {"predictions": predictions,"version": _version}
    print(results)
    #if not errors:

     #   predictions = bike_sharing_pipe.predict(validated_data)
     #   results = {"predictions": predictions,"version": _version, "errors": errors}
        #print(results)

    return results

if __name__ == "__main__":

    data_in={'season':['winter'],'hr':['7am'],'holiday':["No"],'workingday':['Yes'],'weathersit':['Clear'],
                'temp':[26.78],'atemp':[28.9988],'hum':[52.0],'windspeed':[16.9979],'yr':[2012],'mnth':['November']}
    
    make_prediction(input_data=data_in)
