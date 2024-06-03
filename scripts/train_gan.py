import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model, Input
from load_preprocess_data import preprocess_can_data, load_real_can_data
from gan_vae_model import build_generator, build_discriminator

def train_gan(generator, discriminator, combined, data, epochs, batch_size, results_dir):
    for epoch in range(epochs):
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_data = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_data, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))

        print(f"{epoch + 1}/{epochs} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}] [G loss: {g_loss}]")

        if (epoch + 1) % 100 == 0:
            synthetic_data = generator.predict(np.random.normal(0, 1, (1000, latent_dim)))
            pd.DataFrame(synthetic_data).to_csv(f'{results_dir}/synthetic_data_gan_epoch_{epoch + 1}.csv', index=False)

if __name__ == "__main__":
    # Preprocess the raw CAN data (if not already preprocessed)
    preprocess_can_data('../data/can_data.txt')

    # Load the preprocessed real CAN data
    preprocessed_real_can_data = load_real_can_data('../data/preprocessed_can_data.csv')

    latent_dim = 2
    input_dim = preprocessed_real_can_data.shape[1]
    generator = build_generator(latent_dim, input_dim)
    discriminator = build_discriminator(input_dim)
    discriminator.compile(optimizer=Adam(0.0002, 0.5), loss=BinaryCrossentropy(), metrics=['accuracy'])

    z = Input(shape=(latent_dim,))
    fake_data = generator(z)
    discriminator.trainable = False
    validity = discriminator(fake_data)
    combined = Model(z, validity)
    combined.compile(optimizer=Adam(0.0002, 0.5), loss=BinaryCrossentropy())

    train_gan(generator, discriminator, combined, preprocessed_real_can_data, epochs=1000, batch_size=64, results_dir='../results/gan')
