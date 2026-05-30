from src.ML.gaze_prediction_model import GazePredictionModel
from sklearn.linear_model import LogisticRegression
import os
import pickle
import pandas as pd
import json

class GazeLogisticRegModel(GazePredictionModel):

    LR_PARAM_GRID = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga','newton-cg','lbfgs']}
    TRAINING_DATA_SAVE_PATH = os.path.join(GazePredictionModel.WORKING_DIRECTORY, "training_data.csv")
    MODEL_SAVE_PATH = os.path.join(GazePredictionModel.WORKING_DIRECTORY, "lr_model/")
    MODEL_NAME = "lr_model.pkl"
    MODEL_OBJ  = LogisticRegression(max_iter=10000)
    COEF_JSON_PATH = os.path.join(MODEL_SAVE_PATH, "lr_coef_dict.json")


    def __init__(self, target_column=GazePredictionModel.TO_PREDICT, scale=True):
        super().__init__(GazeLogisticRegModel.TRAINING_DATA_SAVE_PATH, GazeLogisticRegModel.MODEL_SAVE_PATH,
                         GazeLogisticRegModel.MODEL_NAME, GazeLogisticRegModel.MODEL_OBJ,
                         GazeLogisticRegModel.LR_PARAM_GRID, GazeLogisticRegModel.COEF_JSON_PATH, 
                         target_column, scale)
        # self.overwrite_coefficients(GazeLogisticRegModel.COEF_JSON_PATH)
        return
    
    # Overwrite write_coefficients to use the weights of the features instead
    def write_coefficients(self, x, coef_json_path):
        MODEL_PATH = os.path.join(GazeLogisticRegModel.MODEL_SAVE_PATH, 
                                  GazeLogisticRegModel.MODEL_NAME)
        with open(MODEL_PATH, "rb") as fp:
            lr_model = pickle.load(fp)

        coef_dict = {}
        i = 0
        for feat in x.columns:
            coef_dict[feat] = lr_model.coef_[0][i]
            i += 1
        coef_dict["intercept"] = str(lr_model.intercept_)

        with open(coef_json_path, "w") as fp:
            json.dump(coef_dict, fp)
        return
    
    # def overwrite_coefficients(self, COEF_JSON_PATH): # MODEL_PATH=os.path.join(MODEL_SAVE_PATH, MODEL_NAME), DATA_PATH = TRAINING_DATA_SAVE_PATH):
    #     MODEL_PATH =os.path.join(GazeLogisticRegModel.MODEL_SAVE_PATH, GazeLogisticRegModel.MODEL_NAME)
    #     with open(MODEL_PATH, "rb") as fp:
    #         lr_model = pickle.load(fp)
    #     with open(COEF_JSON_PATH, "r") as fp:
    #         coef_dict = json.load(fp)

    #     # Now overwrite the values
    #     i = 0
    #     for key in coef_dict:
    #         coef_dict[key] = lr_model.coef_[0][i]
    #         i += 1
    #     coef_dict["intercept"] = str(lr_model.intercept_)
    #     with open(COEF_JSON_PATH, "w") as fp:
    #         json.dump(coef_dict, fp)
    #     # features = df.columns
    #     # for feature, coefficient in zip(features, lr_model.coef_[0]):
    #     #     print(f"{feature}: {coefficient}")
    #     #     coef_dict[feature] = coefficient
    #     # coef_dict["intercept"] = str(lr_model.intercept_)
    #     # with open(COEF_JSON_PATH, 'w') as file:
    #     #     json.dump(coef_dict, file)
    #     return


# class GazePredictionModel:
#     def __init__(self, model_save_path):
#         # Load the model from the pickle file
#         model_path = find_file_with_extension(model_save_path, ".pkl")
#         with open(model_path, 'rb') as file:
#             self.model = pickle.load(file)
#         coef_dict_path = find_file_with_extension(model_save_path, ".json")
#         with open(coef_dict_path, 'r') as file:
#             coef_dict = json.load(file)
#         if "intercept" in coef_dict:
#             coef_dict.pop("intercept")
#         self.feature_names = list(coef_dict.keys())


#     def predict(self, features):
#         # Assume features is a numpy array
#         predictions = self.model.predict(features)
#         # Now 'predictions' contains the predicted classes for the input features
#         return predictions

#     def find_file_with_extension(directory, ext):
#     #return the path of the first file with a given extension string ext under the directory provided.
#         for file in os.listdir(directory):
#             if file.endswith(ext):
#                 return os.path.join(directory, file)
#         return None

# def create_logistic_regression_model(DF_SAVE_PATH, MODEL_DIR):
#     # Define the hyperparameters grid
#     param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
#                 'penalty': ['l1', 'l2'],
#                 'solver': ['liblinear', 'saga','newton-cg','lbfgs']}
    
#     # Get dataframe
#     df = pd.read_csv(DF_SAVE_PATH)

#     x = df.drop(columns=["direction"])
#     #x = df.iloc[:, [0, 2,4 ,11]]
#     y = df["direction"]

#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    
#     logreg = LogisticRegression(max_iter=5000)

#     print("DOING GRID SEARCH...")
#     grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)
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
#     coefficients = best_model.coef_
#     feature_names = x.columns
#     coef_dict = {}
#     for feature, coefficient in zip(feature_names, coefficients[0]):
#         print(f"{feature}: {coefficient}")
#         coef_dict[feature] = coefficient
#     coef_dict["intercept"] = str(best_model.intercept_)

#     import json

#     # Save coef_dict as a JSON file
#     with open(os.path.join(MODEL_DIR,'coef_dict.json'), 'w') as file:
#         json.dump(coef_dict, file)
#     # Now save the model
#     with open(os.path.join(MODEL_DIR,"model.pkl"), 'wb') as file:
#         pickle.dump(best_model, file)
    
#     return best_model


# def test_model(model_save_path, data_path):
#     # Load the model from the pickle file
#     model = GazePredictionModel(model_save_path)
#     df = pd.read_csv(data_path)


#     x = df.drop(columns=["direction"])
#     # Assume x is your test data
#     coefficients = model.model.coef_
#     feature_names = x.columns
#     for feature, coefficient in zip(feature_names, coefficients[0]):
#         print(f"{feature}: {coefficient}")

#     #x = df.iloc[:, [0, 2]]
#     y = df["direction"]
#     # Assume X_test is your test data
#     predictions = model.predict(x)
#     df["predicted_direction"] = predictions
#     # Now 'predictions' contains the predicted classes for the test data
#     from gaze_data_visualization import plot_prediction_vs_actual
#     plot_prediction_vs_actual(df)
#     return df

# def find_file_with_extension(directory, ext):
#     #return the path of the first file with a given extension string ext under the directory provided.
#         for file in os.listdir(directory):
#             if file.endswith(ext):
#                 return os.path.join(directory, file)

# def test_model_from_save_directory(model_save_directory):
#     data_file = find_file_with_extension(model_save_directory,".csv")
#     test_model(model_save_directory, data_file)

# def main():
#     csv_file = "./src/ML/data/test_data_15sec2.csv"
#     model_save_path = "./src/ML/trained_models/lr_model_manual_curated_features_3"
#     test_model("./src/ML/trained_models/lr_model_manual_curated_features_3",csv_file)
#     #create_logistic_regression_model(csv_file, model_save_path)

# if __name__ == '__main__':
#     main()

