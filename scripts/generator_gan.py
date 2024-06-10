import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# Load and preprocess the dataset
data = pd.read_csv('source_dataset.csv')
data = data.fillna(0)  # Fill NaN values with 0
data_values = data[['Timestamp', 'CAN ID', 'DLC'] + [f'DATA[{i}]' for i in range(8)]].values

# Normalize the data
data_min = data_values.min(axis=0)
data_max = data_values.max(axis=0)
data_values = (data_values - data_min) / (data_max - data_min)

# Define the GAN components
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, activation='relu', input_dim=100))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(data_values.shape[1], activation='sigmoid'))
    return model

def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=data_values.shape[1]))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer='adam')

# Define the GAN model
gan_input = layers.Input(shape=(100,))
generated_data = generator(gan_input)
discriminator.trainable = False
gan_output = discriminator(generated_data)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Train the GAN
def train_gan(gan, generator, discriminator, data, epochs=10000, batch_size=32):
    for epoch in range(epochs):
        # Train the discriminator
        noise = np.random.normal(0, 1, (batch_size, 100))
        generated_data = generator.predict(noise)
        real_data = data[np.random.randint(0, data.shape[0], batch_size)]
        combined_data = np.concatenate([generated_data, real_data])
        labels = np.concatenate([np.zeros((batch_size, 1)), np.ones((batch_size, 1))])
        discriminator.trainable = True
        discriminator.train_on_batch(combined_data, labels)
        
        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        misleading_labels = np.ones((batch_size, 1))
        discriminator.trainable = False
        gan.train_on_batch(noise, misleading_labels)

train_gan(gan, generator, discriminator, data_values)

# Generate new synthetic data
def generate_synthetic_data(generator, num_samples):
    noise = np.random.normal(0, 1, (num_samples, 100))
    synthetic_data = generator.predict(noise)
    synthetic_data = synthetic_data * (data_max - data_min) + data_min
    return synthetic_data

synthetic_data_gan = generate_synthetic_data(generator, 1000)

# Convert to DataFrame and save to CSV
synthetic_data_gan_df = pd.DataFrame(synthetic_data_gan, columns=['Timestamp', 'CAN ID', 'DLC'] + [f'DATA[{i}]' for i in range(8)])
synthetic_data_gan_df.to_csv('data/synthetic_data_gan.csv', index=False)