import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from src.ML.gaze_data_visualization import plot_prediction_vs_actual
from sklearn.metrics import make_scorer, f1_score

class GazePredictionModel:

    TO_PREDICT = "direction"
    WORKING_DIRECTORY = "src/ML/trained_models/pipeline_run/"
    SCALE_VALUE = 0.1

    def __init__(self, training_save_path, model_save_path, model_name, model_obj, hyper_param_dict, 
                 coef_json_path, target_column=TO_PREDICT, scale=True):
        # First create the folder
        if not os.path.exists(GazePredictionModel.WORKING_DIRECTORY):
            os.makedirs(GazePredictionModel.WORKING_DIRECTORY)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        # Then do the training
        self.train(training_save_path, model_save_path, model_name, model_obj, 
                hyper_param_dict, coef_json_path, target_column, scale)

        # Load the model from the pickle file
        model_path = self.find_file_with_extension(model_save_path, ".pkl")
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
        coef_dict_path = self.find_file_with_extension(model_save_path, ".json")
        with open(coef_dict_path, 'r') as file:
            coef_dict = json.load(file)
        if "intercept" in coef_dict:
            coef_dict.pop("intercept")
        self.feature_names = list(coef_dict.keys())

    def train(self, training_save_path, model_save_path, model_name, model_obj, 
                hyper_param_dict, coef_json_path, target_column=TO_PREDICT, scale=True):
        # Extract the data used for training and testing
        df = pd.read_csv(training_save_path)
        df = self.remove_outliers(df)
        x = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

        if scale:
            X_train, X_test = self.scale_values(X_train, X_test)

        best_model, test_score = self.grid_search(X_train, y_train, X_test, y_test, model_obj, hyper_param_dict)
        
        # Now save the model
        with open(os.path.join(model_save_path, model_name), 'wb') as file:
            pickle.dump(best_model, file)

        self.write_coefficients(x, coef_json_path)
        return
    
    def grid_search(self, X_train, y_train, X_test, y_test, model_obj, param_grid, cv=5):
        # Use f1-score to reflect class balance/imbalance
        scorer = make_scorer(f1_score, average='macro')
        grid_search = GridSearchCV(estimator=model_obj, param_grid=param_grid, scoring=scorer, cv=cv)
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters and the corresponding accuracy score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print("Best Hyperparameters:", best_params)
        print("Best f1 Score:", best_score)

        # Evaluate the model on the test set
        best_model = grid_search.best_estimator_
        test_score = best_model.score(X_test, y_test)
        print("Test f1 Score:", test_score)

        return best_model, test_score
    
    def remove_outliers(self, df):
        features = [feature for feature in df.columns if feature != GazePredictionModel.TO_PREDICT]

        for feat in features:
            # Calculate Q1 (25th percentile) and Q3 (75th percentile)
            Q1 = df[feat].quantile(0.25)
            Q3 = df[feat].quantile(0.75)
            IQR = Q3 - Q1

            # Identify outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Debug
            print(df[(df[feat] < lower_bound) & (df[feat] > upper_bound)])

            df  = df[(df[feat] >= lower_bound) & (df[feat] <= upper_bound)]
            print(min(df[feat]), max(df[feat]))

        return df
    
    def scale_values(self, X_train, X_test):
            scaler = StandardScaler()
            for feat in X_train.columns:
                if np.abs(X_train[feat].max()- X_train[feat].min()) > GazePredictionModel.SCALE_VALUE:
                    print(f"SCAED: {feat}")
                    X_train[feat] = scaler.fit_transform(X_train[feat].values.reshape(-1, 1)).flatten()
                    X_test[feat] = scaler.transform(X_test[feat].values.reshape(-1, 1)).flatten()
            return X_train, X_test
    
    def test_model(self, model_save_path, data_path):
        # Load the model from the pickle file
        with open(model_save_path, "rb") as fp:
            model = pickle.load(fp)
        df = pd.read_csv(data_path)
        print(df.columns)


        x = df.drop(columns=["direction"])
        # Assume x is your test data
        # coefficients = model.model.coef_
        # feature_names = x.columns
        # for feature, coefficient in zip(feature_names, coefficients[0]):
        #     print(f"{feature}: {coefficient}")

        #x = df.iloc[:, [0, 2]]
        y = df["direction"]
        # Assume X_test is your test data
        predictions = model.predict(x)
        # for val in predictions:
        #     print(val)
        df["predicted_direction"] = predictions
        # Now 'predictions' contains the predicted classes for the test data
        plot_prediction_vs_actual(df)
        return df
    
    def write_coefficients(self, df, COEF_JSON_PATH):
        with open(COEF_JSON_PATH, "w") as fp:
            coeff_dict = {}
            for col in df.columns:
                coeff_dict[col] = True
            json.dump(coeff_dict, fp)
        return

    def predict(self, features):
        # Assume features is a numpy array
        predictions = self.model.predict(features)
        # Now 'predictions' contains the predicted classes for the input features
        return predictions

    def find_file_with_extension(self, directory, ext):
        #return the path of the first file with a given extension string ext under the directory provided.
        for file in os.listdir(directory):
            if file.endswith(ext):
                return os.path.join(directory, file)
        return None