# dataset-generator

## Installing all the necessary packages

```
python3 -m pip install numpy pandas matplotlib scikit-learn tensorflow
```

## Running the Scripts

### 1: Run the preprocessing script

```
python3 scripts/preprocess.py
```

### 2: Train and generate using the GAN model

```
python3 scripts/generator_gan.py
```

### 3: Train and generate using the VAE model

```
python3 scripts/generator_vae.py
```

### 4: Train and generate using the GAN-VAE model

```
python3 scripts/generator_gan_vae.py
```

### 5: Validate and compare the synthetic data
