import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

from bike_sharing_model.config.core import config
from bike_sharing_model.pipeline import bike_sharing_pipeline
# from bike_sharing_model.predict import make_prediction
from bike_sharing_model.processing.data_manager import load_dataset, save_pipeline



def run_training() -> None:
    data = load_dataset(file_name=config.app_config.training_data_file)

    X = data.drop(config.model_config.target, axis=1)
    y = data[config.model_config.target]
    print(f"Data used for FIT METHOD: {X.columns}")
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X,  # predictors
        y,
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )

    # Pipeline fitting
    bike_sharing_pipeline.fit(X_train, y_train)
    y_pred = bike_sharing_pipeline.predict(X_test)
    # print("Accuracy(in %):", accuracy_score(y_test, y_pred)*100)

    print("R2 score:", r2_score(y_test, y_pred))
    print("Mean squared error:", mean_squared_error(y_test, y_pred))

    # persist trained model
    save_pipeline(pipeline_to_persist= bike_sharing_pipeline)
    # printing the score
    
if __name__ == "__main__":
    run_training()

    #bike_sharing_model\predict.py