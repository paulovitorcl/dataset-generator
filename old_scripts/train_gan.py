import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam

def build_generator(latent_dim, output_dim):
    model = Sequential()
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

def build_discriminator(input_dim):
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

def train_gan(data, epochs=1000, batch_size=64, latent_dim=100):
    input_dim = data.shape[1]

    # Normalize data to [-1, 1]
    data = (data - data.min()) / (data.max() - data.min())  # Min-Max normalization to [0, 1]
    data = data * 2 - 1  # Normalize to [-1, 1]

    generator = build_generator(latent_dim, input_dim)
    discriminator = build_discriminator(input_dim)

    optimizer = Adam(0.0002, 0.5)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    z = tf.keras.Input(shape=(latent_dim,))
    generated_data = generator(z)
    discriminator.trainable = False
    validity = discriminator(generated_data)

    combined = tf.keras.Model(z, validity)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_data = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_data, real)
        d_loss_fake = discriminator.train_on_batch(gen_data, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = combined.train_on_batch(noise, real)

        if epoch % 1000 == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | D acc.: {100 * d_loss[1]}] [G loss: {g_loss}]")
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}")

    return generator

if __name__ == "__main__":
    # Load data
    data = pd.read_csv('data/preprocessed_can_data.csv').values[:, 3:]  # Skip non-data columns
    generator = train_gan(data)
    
    # Generate data
    noise = np.random.normal(0, 1, (1000, 100))
    generated_data = generator.predict(noise)
    generated_data = (generated_data + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
    generated_data = generated_data * (data.max() - data.min()) + data.min()  # Reverse min-max normalization
    generated_data = np.round(generated_data).astype(int)  # Round to integers

    pd.DataFrame(generated_data).to_csv('results/gan/generated_data.csv', index=False)
