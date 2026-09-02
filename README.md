# Debiasing Variational Autoencoder

 PyTorch reimplementation of Uncovering and Mitigating Algorithmic Bias through Learned Latent Structure (Amini et al., AAAI 2019)

 A DB-VAE learns latent structure over face images with a VAE branch attached to a supervised classifier. During training, it adjusts the sampling weights of face examples according to the distribution of their learned latent variables, increasing the sampling probability of underrepresented regions in latent space.

The method does not require demographic labels during training. Instead, it uses learned latent structure to identify rare variations within the face class. These variations may correlate with demographic attributes such as skin tone or gender.

The goal is therefore not to solve the face/non-face class imbalance itself. Class balance is handled by balanced batching. This algorithm addresses the intra-class imbalance confronted as underrepresented variation within the face class in papers example


## Files

```text
models.py       encoder, decoder, DBVAE, StandardCNN, reparameterize
losses.py       KL, L1 reconstruction, DB-VAE conditional loss
sampler.py      balanced batching and paper adaptive resampling
data.py        CelebA/non-face training loader and PPB evaluation loader
train.py        train DB-VAE or the CNN baseline
evaluate.py     evaluate a saved checkpoint on PPB or fallback folders
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```



## Training

CNN:

```bash
python train.py --baseline \
    --celeba-dir data/celeba/img_align_celeba \
    --nonface-dir data/imagenet_nonface \
    --output runs/baseline
```

DB-VAE:

```bash
python train.py \
    --celeba-dir data/celeba/img_align_celeba \
    --nonface-dir data/imagenet_nonface \
    --output runs/dbvae
```

The default training setup uses 64×64 RGB images, balanced face/non-face batches, latent dimension 100, a KL weight of `5e-4`, and adaptive face resampling refreshed once per epoch.

## Evaluation

PPB:

```bash
python evaluate.py --model runs/dbvae/model.pt \
    --ppb-dir data/ppb \
    --ppb-csv data/ppb/annotations.csv
```





## Reference

Amini, A., Soleimany, A., Schwarting, W., Bhatia, S., Rus, D. *Uncovering and Mitigating Algorithmic Bias through Learned Latent Structure.* AAAI, 2019.
