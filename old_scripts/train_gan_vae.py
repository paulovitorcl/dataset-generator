import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, LeakyReLU, BatchNormalization
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

def build_generator(latent_dim, output_dim):
    model = tf.keras.Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(output_dim, activation='tanh'))
    return model

def train_gan_vae(data, latent_dim=2, epochs=10000, batch_size=64, gan_latent_dim=100):
    data = (data - 0.5) * 2.0  # Normalize to [-1, 1]
    input_dim = data.shape[1]

    # VAE
    vae, encoder, decoder = build_vae(input_dim, latent_dim)
    vae.compile(optimizer='adam')
    vae.fit(data, epochs=50, batch_size=batch_size)

    # GAN
    generator = build_generator(gan_latent_dim, input_dim)
    discriminator = Sequential([
        Dense(512, input_dim=input_dim),
        LeakyReLU(alpha=0.2),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')
    ])
    discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    z = tf.keras.Input(shape=(gan_latent_dim,))
    generated_data = generator(z)
    discriminator.trainable = False
    validity = discriminator(generated_data)
    combined = tf.keras.Model(z, validity)
    combined.compile(loss='binary_crossentropy', optimizer='adam')

    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx]

        noise = np.random.normal(0, 1, (batch_size, gan_latent_dim))
        gen_data = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_data, real)
        d_loss_fake = discriminator.train_on_batch(gen_data, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, gan_latent_dim))
        g_loss = combined.train_on_batch(noise, real)

        if epoch % 1000 == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | D acc.: {100*d_loss[1]}] [G loss: {g_loss}]")

    return vae, encoder, decoder, generator

if __name__ == "__main__":
    data = pd.read_csv('data/preprocessed_can_data.csv').values
    vae, encoder, decoder, generator = train_gan_vae(data)
    
    # Generate using VAE
    z_sample = np.random.normal(size=(1000, 2))
    vae_generated_data = decoder.predict(z_sample)
    pd.DataFrame(vae_generated_data).to_csv('results/gan_vae/generated_vae_data.csv', index=False)
    
    # Generate using GAN
    noise = np.random.normal(0, 1, (1000, 100))
    gan_generated_data = generator.predict(noise)
    gan_generated_data = (gan_generated_data + 1) / 2.0  # Denormalize to [0, 1]
    pd.DataFrame(gan_generated_data).to_csv('results/gan_vae/generated_gan_data.csv', index=False)
