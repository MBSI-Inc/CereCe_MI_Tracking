import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_data(csv_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Extract the first two columns as x and y axes
    x = df.iloc[:, 0]
    y = df.iloc[:, 2]

    # Extract the label column
    labels = df.iloc[:, -1]

    # Create a scatter plot
    plt.scatter(x, y, c=labels)

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot')



def pairplot(csv_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Extract the label column
    labels = df.iloc[:, -1]

    # Drop the label column from the DataFrame
    df = df.iloc[:, :-1]

    # Create a pairplot
    
    axes = pd.plotting.scatter_matrix(df, c=labels, figsize=(10, 10))
    for ax in axes.flatten():
        ax.xaxis.label.set_rotation(90)
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha('right')


    # Show the plot
    plt.show()

def plot_prediction_vs_actual(df):
    # Extract the actual and predicted labels
    actual = df['direction']
    predictions = df['predicted_direction']

    # Create a scatter plot
    # Add perturbation to actual values
    actual_purturbed = actual + np.random.normal(0, 0.1, len(actual))
    predictions_purdurbed = predictions + np.random.normal(0, 0.1, len(predictions))

    # Create a scatter plot with perturbed actual values
    plt.scatter(actual_purturbed, predictions_purdurbed, c = actual, alpha= 0.3)
    # plt.scatter(actual, predictions, c = actual, alpha= 0.3)

    
    # Add labels and title
    plt.xlabel('Direction')
    plt.ylabel('Predicted direction')
    plt.title('Direction vs Predicted direction')

    # Show the plot
    plt.show()

def main():
    csv_file = "./src/ML/data/test_data_15sec2.csv"
    pairplot(csv_file)

if __name__ == '__main__':
    main()
