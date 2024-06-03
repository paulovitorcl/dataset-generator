# dataset-generator

## Installing all the necessary packages

```
python3 -m pip install numpy pandas matplotlib scikit-learn tensorflow
```

## Running the Scripts

### 1: Run the preprocessing script

```
python3 scripts/load_preprocess_data.py
```

### 2: Train the GAN model

```
python3 scripts/train_gan.py
```

### 3: Train the VAE model

```
python3 scripts/train_vae.py
```

### 4: Train the GAN-VAE model

```
python3 scripts/train_gan_vae.py
```

### 5: Validate and compare the synthetic data:

```
python3 scripts/validate.py
```
