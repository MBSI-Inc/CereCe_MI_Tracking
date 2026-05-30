import sys, os
# sys.path.append("./src/ML/")
# from logistic_reg import create_logistic_regression_model, GazePredictionModel, test_model
from src.ML.model_direction_tester import test_model_realtime
from facemesh_gazetracker import GazeTracker
from src.ML.logistic_reg import GazeLogisticRegModel
from src.ML.neural_net import GazeNeuralNetModel
from src.ML.svm import GazeSVMModel
from src.ML.dataCollector import collect_data

LIST_OF_MODELS = ["neural_net", "logistic_reg", "svm"]

def main():
    # 0) Ensure the directory actually exists
    WORKING_DIRECTORY = "src/ML/trained_models/pipeline_run"
    if not os.path.exists(WORKING_DIRECTORY):
        os.makedirs(WORKING_DIRECTORY)

    # 1) identify which model to use and start training
    model_class = select_model(LIST_OF_MODELS)
    collect_data(model_class.TRAINING_DATA_SAVE_PATH)
    model_instance = model_class()

    # 2) Test the model's results
    model_instance.test_model(os.path.join(model_instance.MODEL_SAVE_PATH,model_instance.MODEL_NAME), 
                              model_instance.TRAINING_DATA_SAVE_PATH)

    # 3) Test the model in real time
    gaze_tracker = GazeTracker("config.json")
    data = test_model_realtime(gaze_tracker, model_instance)
    return

def select_model(list_of_options):
    model_selected = input(create_input_msg(list_of_options))

    # Add models here based on the text
    if model_selected == "neural_net":
        model = GazeNeuralNetModel
    elif model_selected == "svm":
        model = GazeSVMModel
    elif model_selected == "logistic_reg":
        model = GazeLogisticRegModel
    else:
        raise ValueError("Expected a recognized model, got an unknown one")

    return model

def create_input_msg(list_of_options):
    # First the header message
    text = "Choose a model below by typing in one of the following names:\n"
    for model in list_of_options:
        text += f"- {model}\n"
    text += "Your Desired Model: "
    return text

if __name__ == "__main__":
    main()

# WORKING_DIRECTORY = "/trained_models/pipeline_run"
# TRAINING_DATA_SAVE_PATH = os.path.join(WORKING_DIRECTORY, "training_data.csv")

# ### directory to save the model
# if not os.path.exists(WORKING_DIRECTORY):
#     os.makedirs(WORKING_DIRECTORY)

# MODEL_SAVE_DIR = os.path.join(WORKING_DIRECTORY, "model/")
# # Create the directory based on MODEL_SAVE_DIR if it doesn't exist
# if not os.path.exists(MODEL_SAVE_DIR):
#     os.makedirs(MODEL_SAVE_DIR)

# # Generate the training data with data collector and save to a .csv file
# collect_data(TRAINING_DATA_SAVE_PATH)

# # Create the logistic regression model using the .csv file and save to model_save_path
# create_logistic_regression_model(TRAINING_DATA_SAVE_PATH, MODEL_SAVE_DIR)

# # test the model. In that function you can see example of how to load the model and predict from features.
# test_model(MODEL_SAVE_DIR, TRAINING_DATA_SAVE_PATH)

# # example to load the model
# gaze_model = GazePredictionModel(MODEL_SAVE_DIR)
# gaze_tracker = GazeTracker("config.json")
# data = test_model_realtime(gaze_tracker, gaze_model)