import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from load_preprocess_data import preprocess_can_data, load_real_can_data
from gan_vae_model import build_vae

def train_vae(vae, data, epochs, batch_size, results_dir):
    vae.compile(optimizer=Adam(), loss=None)
    vae.fit(data, data, epochs=epochs, batch_size=batch_size)
    z_mean, z_log_var, z = vae.get_layer('encoder').predict(data)

    pd.DataFrame(z).to_csv(f'{results_dir}/latent_space.csv', index=False)
    print(f"Latent space representations saved to {results_dir}/latent_space.csv")

if __name__ == "__main__":
    # Preprocess the raw CAN data (if not already preprocessed)
    preprocess_can_data('data/can_data.txt')

    # Load the preprocessed real CAN data
    preprocessed_real_can_data = load_real_can_data('data/preprocessed_can_data.csv')

    latent_dim = 2
    input_dim = preprocessed_real_can_data.shape[1]
    vae, encoder, decoder = build_vae(input_dim, latent_dim)

    train_vae(vae, preprocessed_real_can_data, epochs=1000, batch_size=64, results_dir='results/vae')
