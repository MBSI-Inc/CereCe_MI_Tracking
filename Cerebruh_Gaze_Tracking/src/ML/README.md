# ONBOARDING FOR ML PROJECT

- The goal of the gaze-tracking team is to create a reliable program that can (for now) tell a wheelchair to keep going straight, turn left, or turn right based on where a person looks. 

- The machine learing portion of this project attempts to create a classifier model that uses the different features taken from our program (iris coordinates, head rotation, etc.) to predict the direction a person wants to go towards. This involves 2 main steps:


## Data Collection Step

-  This is contained in  the  `dataCollector.py` file and is then parsed using the feature extractor class and functions in the `feature_extractor` file. Directions are stored as 0 for straight, 1 for right, and -1 for left.


## Training Step
- We use the collected data to train and test our models. This step is broken down into different files with designated functions
        - `logistic_reg`, `neural_net`, `svm`: These files run the entire training and testing process using the collected data for a logistic regression, neural network (MLP), and support vector machine (svm) classifiers respectively.
        - `test_model` : This is a function present within all the previously mentioned files. It uses the `plot_prediction_vs_actual` function from the `gaze_data_visualization` file to plot predicted versus actual values.
        - `test_model_realtime`: This is a live test of the classifier. A window pops up with your camera view and a square. The square goes the direction of where the classifier thinks your desired direction is.
        - `trained_models`: This folder stores all the models and training data. In `pipeline_run`, models are stored in different folders (model, nn_model, svm_model for a logistic regression, neural network, and svm model). Each folder has the designated model and a json file contianing the features used to train the model.

## Onboarding Tasks
- First, let's try run the `pipeline.py` file. This runs the entire data collection and training process for a logistic regression classifier.
    - The first step is to collect data. You will be prompted to look left, right, and center for a set amount of time. You can check if the data has been collected if your `pipeline_run` folder in your `trained_models` folder contains a `training_data.csv` file.
    - The second step is to do the actual training. This will be instantaneous with both the logistic regression and svm classifiers.
        - The step will finish when a window pops up. This is a predicted-versus-actual values plot that pops up for the different values. There is a bit of randomized noise added to the points so they look more like clumps (easier to compare accuracy) than just a single dot (since all the values will either be -1, 0, or 1 anyway)
        - Two files are stored
            - A `json` file containing the coefficients used to train the model
            - A `pkl` file containing the model
    - Once you close this plot window, `test_model_realtime` runs. You can look to the left or the right as you did during the data collection phase to check if the model follows.

- After trying these files, we can try `neural_net` and `svm` files. Running them is similar to `pipeline_run`, where new folders are created for different models while overwriting the `training_data.csv` file.

- Once you've done these things, you're up to date with the current modelling progress.