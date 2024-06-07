import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.losses import mse
import tensorflow.keras.backend as K

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

class VAELossLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, **kwargs):
        self.input_dim = input_dim
        super(VAELossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        x, x_decoded_mean, z_mean, z_log_var = inputs
        reconstruction_loss = mse(x, x_decoded_mean)
        reconstruction_loss *= self.input_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.add_loss(vae_loss)
        return x_decoded_mean

def build_vae(input_dim, latent_dim):
    inputs = Input(shape=(input_dim,))
    h = Dense(256, activation='relu')(inputs)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    h_decoded = Dense(256, activation='relu')(latent_inputs)
    outputs = Dense(input_dim, activation='sigmoid')(h_decoded)

    decoder = Model(latent_inputs, outputs, name='decoder')
    outputs = decoder(encoder(inputs)[2])

    vae_loss_layer = VAELossLayer(input_dim)([inputs, outputs, z_mean, z_log_var])
    vae = Model(inputs, vae_loss_layer, name='vae_mlp')

    return vae, encoder, decoder

def train_vae(data, latent_dim=2, epochs=50, batch_size=32):
    input_dim = data.shape[1]
    vae, encoder, decoder = build_vae(input_dim, latent_dim)
    vae.compile(optimizer='adam')
    vae.fit(data, data, epochs=epochs, batch_size=batch_size)  # Fit VAE to data
    return vae, encoder, decoder

if __name__ == "__main__":
    # Load data
    data = pd.read_csv('data/preprocessed_can_data.csv').values[:, 3:]  # Skip non-data columns

    # Normalize data to [0, 1]
    data = data.astype('float32')
    data_min = data.min()
    data_max = data.max()
    data = (data - data_min) / (data_max - data_min)

    # Train VAE
    vae, encoder, decoder = train_vae(data)

    # Generate new data
    z_sample = np.random.normal(size=(1000, 2))
    generated_data = decoder.predict(z_sample)

    # Denormalize generated data
    generated_data = generated_data * (data_max - data_min) + data_min
    generated_data = np.round(generated_data).astype(int)

    # Save generated data
    pd.DataFrame(generated_data).to_csv('results/vae/generated_data.csv', index=False)
