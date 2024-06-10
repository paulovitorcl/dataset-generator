import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K

# Load and preprocess the dataset
data = pd.read_csv('data/source_dataset.csv')
data = data.fillna(0)  # Fill NaN values with 0
data_values = data[['Timestamp', 'CAN ID', 'DLC'] + [f'DATA[{i}]' for i in range(8)]].values

# Normalize the data
data_min = data_values.min(axis=0)
data_max = data_values.max(axis=0)
data_values = (data_values - data_min) / (data_max - data_min)

# Define the VAE components
latent_dim = 2
input_shape = data_values.shape[1]

# Sampling function for VAE
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Encoder
inputs = tf.keras.Input(shape=(input_shape,))
x = layers.Dense(128, activation='relu')(inputs)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)
z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Decoder
latent_inputs = tf.keras.Input(shape=(latent_dim,))
x = layers.Dense(128, activation='relu')(latent_inputs)
outputs = layers.Dense(input_shape, activation='sigmoid')(x)

encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')
decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')

# VAE model
outputs = decoder(encoder(inputs)[2])
vae = tf.keras.Model(inputs, outputs, name='vae')

# VAE loss
reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs) * input_shape
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# Train the VAE
vae.fit(data_values, epochs=100, batch_size=32)

# Generate new synthetic data
def generate_synthetic_data_vae(decoder, num_samples):
    z_sample = np.random.normal(size=(num_samples, latent_dim))
    synthetic_data = decoder.predict(z_sample)
    synthetic_data = synthetic_data * (data_max - data_min) + data_min
    return synthetic_data

synthetic_data_vae = generate_synthetic_data_vae(decoder, 1000)

# Convert to DataFrame and save to CSV
synthetic_data_vae_df = pd.DataFrame(synthetic_data_vae, columns=['Timestamp', 'CAN ID', 'DLC'] + [f'DATA[{i}]' for i in range(8)])
synthetic_data_vae_df.to_csv('synthetic_data_vae.csv', index=False)
