import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle
import os, json
from src.ML.model_direction_tester import test_model_realtime
from src.ML.gaze_prediction_model import GazePredictionModel
from facemesh_gazetracker import GazeTracker
from src.ML.dataCollector import collect_data

class GazeNeuralNetModel(GazePredictionModel):
    NN_PARAM_GRID = {"activation": ["identity", "logistic", "tanh", "relu"],
                'learning_rate_init' : [0.0001, 0.001, 0.01, 0.1],
                'solver': ['sgd','adam','lbfgs'],
                'learning_rate' : ['constant', 'invscaling', 'adaptive']
    }
    TRAINING_DATA_SAVE_PATH = os.path.join(GazePredictionModel.WORKING_DIRECTORY, "training_data.csv")
    MODEL_SAVE_PATH = os.path.join(GazePredictionModel.WORKING_DIRECTORY, "nn_model/")
    MODEL_NAME = "nn_model.pkl"
    MODEL_OBJ  = MLPClassifier(max_iter=5000)
    COEF_JSON_PATH = os.path.join(MODEL_SAVE_PATH, "nn_coef_dict.json")
    
    def __init__(self, target_column=GazePredictionModel.TO_PREDICT, scale=True):
        super().__init__(GazeNeuralNetModel.TRAINING_DATA_SAVE_PATH, GazeNeuralNetModel.MODEL_SAVE_PATH,
                         GazeNeuralNetModel.MODEL_NAME, GazeNeuralNetModel.MODEL_OBJ,
                         GazeNeuralNetModel.NN_PARAM_GRID, GazeNeuralNetModel.COEF_JSON_PATH, 
                         target_column, scale)
        return

# def create_nn_classifier(DF_SAVE_PATH, MODEL_DIR, COEF_PATH):
#     param_grid = {"activation": ["identity", "logistic", "tanh", "relu"],
#                 'learning_rate_init' : [0.0001, 0.001, 0.01, 0.1],
#                 'solver': ['sgd','adam','lbfgs'],
#                 'learning_rate' : ['constant', 'invscaling', 'adaptive']
#     }

#     # Get dataframe
#     df = pd.read_csv(DF_SAVE_PATH)
#     print(df.columns)

#     x = df.drop(columns=["direction"])
#     #x = df.iloc[:, [0, 2,4 ,11]]
#     y = df["direction"]
#     write_coefficients(x, COEF_PATH)

#     # Split the data into training and testing sets
#     scaler = StandardScaler()
#     X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
    
#     nn = MLPClassifier(max_iter=5000)

#     print("DOING GRID SEARCH...")
#     grid_search = GridSearchCV(estimator=nn, param_grid=param_grid, cv=5)
#     grid_search.fit(X_train, y_train)

#     # Get the best hyperparameters and the corresponding accuracy score
#     best_params = grid_search.best_params_
#     best_score = grid_search.best_score_

#     print("Best Hyperparameters:", best_params)
#     print("Best Accuracy Score:", best_score)

#     # Evaluate the model on the test set
#     best_model = grid_search.best_estimator_
#     test_score = best_model.score(X_test, y_test)
#     print("Test Accuracy Score:", test_score)
#     # Now save the model
#     with open(os.path.join(MODEL_DIR,"nn_model.pkl"), 'wb') as file:
#         pickle.dump(best_model, file)
    
#     return best_model

# def test_model(model_save_path, data_path):
#     # Load the model from the pickle file
#     with open(model_save_path, 'rb') as fp:
#         model = pickle.load(fp)
#     df = pd.read_csv(data_path)
#     print(df.columns)


#     x = df.drop(columns=["direction"])
#     # Assume x is your test data
#     # coefficients = model.model.coef_
#     # feature_names = x.columns
#     # for feature, coefficient in zip(feature_names, coefficients[0]):
#     #     print(f"{feature}: {coefficient}")

#     #x = df.iloc[:, [0, 2]]
#     y = df["direction"]
#     # Assume X_test is your test data
#     predictions = model.predict(x)
#     # for val in predictions:
#     #     print(val)
#     df["predicted_direction"] = predictions
#     # Now 'predictions' contains the predicted classes for the test data
#     from gaze_data_visualization import plot_prediction_vs_actual
#     plot_prediction_vs_actual(df)
#     return df

# def write_coefficients(df, COEF_JSON_PATH):
#     with open(COEF_JSON_PATH, "w") as fp:
#         coeff_dict = {}
#         for col in df.columns:
#             coeff_dict[col] = True
#         json.dump(coeff_dict, fp)
#     return

# def main():
#     WORKING_DIRECTORY = "src/ML/trained_models/pipeline_run/"
#     NN_DIRECTORY = os.path.join(WORKING_DIRECTORY, "nn_model/")
#     TRAINING_DATA_SAVE_PATH = os.path.join(WORKING_DIRECTORY, "training_data.csv")
#     if not os.path.exists(NN_DIRECTORY):
#         os.makedirs(NN_DIRECTORY)
#     NN_COEF_PATH = os.path.join(NN_DIRECTORY, "nn_coef_dict.json")
    
#     # Now train the model
#     collect_data(TRAINING_DATA_SAVE_PATH)
#     create_nn_classifier(TRAINING_DATA_SAVE_PATH, NN_DIRECTORY, NN_COEF_PATH)
#     NN_MODEL_PATH = os.path.join(NN_DIRECTORY, "nn_model.pkl")
#     test_model(NN_MODEL_PATH, TRAINING_DATA_SAVE_PATH)
#     gaze_model = GazePredictionModel(NN_DIRECTORY)
#     gaze_tracker = GazeTracker("config.json")
#     test_model_realtime(gaze_tracker, gaze_model)
#     return

# if __name__ == "__main__":
#     main()