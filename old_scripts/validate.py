import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from load_preprocess_data import load_real_can_data

def plot_distribution(real_data, synthetic_data, method):
    num_bins = 50
    plt.figure(figsize=(10, 6))

    plt.hist(real_data, num_bins, alpha=0.5, label='Real Data', color='blue')
    plt.hist(synthetic_data, num_bins, alpha=0.5, label='Synthetic Data', color='orange')
    plt.title(f'Distribution of CAN Data - {method}')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.savefig(f'distribution_{method}.png')
    plt.show()

def validate(real_data_file, synthetic_data_file, method):
    # Load real and synthetic data
    real_data = load_real_can_data(real_data_file)
    synthetic_data = pd.read_csv(synthetic_data_file).values

    # Calculate mean squared error between real and synthetic data
    mse = mean_squared_error(real_data, synthetic_data)
    print(f"Mean Squared Error between real and synthetic data ({method}): {mse}")

    # Plot distribution comparison
    plot_distribution(real_data.flatten(), synthetic_data.flatten(), method)

if __name__ == "__main__":
    real_data_file = 'data/preprocessed_can_data.csv'
    results_dir = 'results/'

    scenarios = ['gan', 'vae', 'gan_vae']

    for scenario in scenarios:
        synthetic_data_file = f'{results_dir}/{scenario}/synthetic_data_{scenario}_epoch_1000.csv'
        validate(real_data_file, synthetic_data_file, scenario)
