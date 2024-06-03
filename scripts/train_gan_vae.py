import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model, Input
from load_preprocess_data import preprocess_can_data, load_real_can_data
from gan_vae_model import build_vae, build_generator, build_discriminator

def train_gan_vae(vae, generator, discriminator, combined, data, epochs, batch_size, results_dir):
    # Train the VAE first
    vae.compile(optimizer=Adam(), loss=None)
    vae.fit(data, data, epochs=epochs, batch_size=batch_size)

    # Generate latent space representations using VAE's encoder
    z_mean, z_log_var, z = vae.get_layer('model_1').predict(data)

    for epoch in range(epochs):
        idx = np.random.randint(0, data.shape[0], batch_size)
        latent_real = z[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_data = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(latent_real, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_data, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))

        print(f"{epoch + 1}/{epochs} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}] [G loss: {g_loss}]")

        if (epoch + 1) % 100 == 0:
            synthetic_data = generator.predict(np.random.normal(0, 1, (1000, latent_dim)))
            pd.DataFrame(synthetic_data).to_csv(f'{results_dir}/synthetic_data_gan_vae_epoch_{epoch + 1}.csv', index=False)

if __name__ == "__main__":
    # Preprocess the raw CAN data (if not already preprocessed)
    preprocess_can_data('../data/can_data.txt')

    # Load the preprocessed real CAN data
