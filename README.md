# Debiasing Variational Autoencoder

 PyTorch reimplementation of Uncovering and Mitigating Algorithmic Bias through Learned Latent Structure (Amini et al., AAAI 2019)

 A DB-VAE learns latent structure over face images with a VAE branch attached to a supervised classifier. During training, it adjusts the sampling weights of face examples according to the distribution of their learned latent variables, increasing the sampling probability of underrepresented regions in latent space.

The method does not require demographic labels during training. Instead, it uses learned latent structure to identify rare variations within the face class. These variations may correlate with demographic attributes such as skin tone or gender.

The goal is therefore not to solve the face/non-face class imbalance itself. Class balance is handled by balanced batching. This algorithm addresses the intra-class imbalance confronted as underrepresented variation within the face class in papers example.

The training is done with images for positive class (face) from CelebA and negative class (non-face) ImageNet datasets and the evaluation of the algorithm to measure the variations within the face class is done using PPB (Pilot Parliaments Benchmark)

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


## Evaluation

PPB:

```bash
python evaluate.py --model runs/dbvae/model.pt \
    --ppb-dir data/ppb \
    --ppb-csv data/ppb/annotations.csv
```

## Data

The datasets (CelebA, ImageNet and PPB for demographic evaluation) are not included in this repository.

After downloading the datasets, place them in the following structure:

```text
data/
├── celeba/
│   └── img_align_celeba/
├── imagenet_nonface/
└── ppb/
    ├── images/
    └── annotations.csv
```


## Reference

Amini, A., Soleimany, A., Schwarting, W., Bhatia, S., Rus, D. *Uncovering and Mitigating Algorithmic Bias through Learned Latent Structure.* AAAI, 2019.
