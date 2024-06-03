import os

if __name__ == "__main__":
    if not os.path.exists('../results/gan'):
        os.makedirs('../results/gan')
    if not os.path.exists('../results/vae'):
        os.makedirs('../results/vae')
    if not os.path.exists('../results/gan_vae'):
        os.makedirs('../results/gan_vae')

    os.system('python train_gan.py')
    os.system('python train_
