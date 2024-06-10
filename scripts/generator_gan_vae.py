import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

# Load and preprocess the dataset
data = pd.read_csv('data/source_dataset.csv')
data = data.fillna(0)  # Fill NaN values with 0
data_values = data[['Timestamp', 'CAN ID', 'DLC'] + [f'DATA[{i}]' for i in range(8)]].values

# Normalize the data
data_min = data_values.min(axis=0)
data_max = data_values.max(axis=0)
data_values = (data_values - data_min) / (data_max - data_min)

# Define the GAN-VAE components
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

# Generator (Decoder)
latent_inputs = tf.keras.Input(shape=(latent_dim,))
x = layers.Dense(128, activation='relu')(latent_inputs)
outputs = layers.Dense(input_shape, activation='sigmoid')(x)

encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')
generator = tf.keras.Model(latent_inputs, outputs, name='generator')

# Discriminator
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=input_shape))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer='adam')

# GAN-VAE model
outputs = generator(encoder(inputs)[2])
gan_vae = tf.keras.Model(inputs, outputs, name='gan_vae')

# GAN-VAE loss
reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs) * input_shape
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
gan_vae.add_loss(vae_loss)
gan_vae.compile(optimizer='adam')

# Train the GAN-VAE
gan_vae.fit(data_values, epochs=100, batch_size=32)

# Generate new synthetic data
def generate_synthetic_data_gan_vae(generator, num_samples):
    z_sample = np.random.normal(size=(num_samples, latent_dim))
    synthetic_data = generator.predict(z_sample)
    synthetic_data = synthetic_data * (data_max - data_min) + data_min
    return synthetic_data

synthetic_data_gan_vae = generate_synthetic_data_gan_vae(generator, 1000)

# Convert to DataFrame and save to CSV
synthetic_data_gan_vae_df = pd.DataFrame(synthetic_data_gan_vae, columns=['Timestamp', 'CAN ID', 'DLC'] + [f'DATA[{i}]' for i in range(8)])
synthetic_data_gan_vae_df.to_csv('data/synthetic_data_gan_vae.csv', index=False)
