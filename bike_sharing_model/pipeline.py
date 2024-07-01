import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from bike_sharing_model.config.core import config
from bike_sharing_model.processing.features import WeathersitImputer
from bike_sharing_model.processing.features import Mapper

bike_sharing_pipeline = Pipeline([

    ('weathersit_imputation', WeathersitImputer(variable=config.model_config.weathersit_var)),

    ('map_year', Mapper(config.model_config.year_var, config.model_config.year_mapping)),
    ('map_mnth', Mapper(config.model_config.mnth_var, config.model_config.mnth_mapping)),
    ('map_season', Mapper(config.model_config.season_var, config.model_config.season_mapping)),
    ('map_weather', Mapper(config.model_config.weathersit_var, config.model_config.weather_mapping)),
    ('map_holiday', Mapper(config.model_config.holiday_var, config.model_config.holiday_mapping)),
    ('map_workingday', Mapper(config.model_config.workingday_var, config.model_config.workingday_mapping)),
    ('map_hr', Mapper(config.model_config.hr_var, config.model_config.hr_mapping)),

    ('scaler', StandardScaler()),

    ('model_rf', RandomForestRegressor(n_estimators=config.model_config.n_estimators,
                                       max_depth=config.model_config.max_depth,
                                       random_state=config.model_config.random_state))

])
    #pipeline_list = [

        #('weekday_imputer', WeekdayImputer('weekday')), #unused feature; lets delete later
        #('weathersit_imputer', WeathersitImputer('weathersit')),
        #('yr_mapper', Mapper('yr', config.model_config.map_yr)),
        #('mnth_mapper', Mapper('mnth', config.model_config.mnth_mapping)),
        #('season_mapper', Mapper('season', config.model_config.season_mapping)),
        #('weather_mapper', Mapper('weathersit', config.model_config.weather_mapping)),
        #('holiday_mapper', Mapper('holiday', config.model_config.holiday_mapping)),
        #('workingday_mapper', Mapper('workingday', config.model_config.workingday_mapping)),
        #('hour_mapper', Mapper('hr', config.model_config.hour_mapping)),
        #('weekday_ohe', WeekdayOneHotEncoder('weekday')), #unused feature; lets delete later
        #("scaler", StandardScaler())
        #]

    # Create the pipeline object
    #pipe = Pipeline(pipeline_list)

    # Define the final regressor model
    #reg = RandomForestRegressor()
    #reg = linear_model.LinearRegression()

    # Add the regressor to the pipeline
    #pipeline_list.append(('reg', reg))

    # Create the final pipeline object
    #pipe = Pipeline(pipeline_list)

    # Print the pipeline
    #return pipe
