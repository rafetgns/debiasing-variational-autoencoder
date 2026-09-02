# Train DB-VAE or a  CNN baseline on CelebA + a non-face folder

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import CelebAImageNetDataset
from losses import dbvae_loss
from models import DBVAE, StandardCNN
from sampler import BalancedBatchSampler, compute_sample_weights, get_latent_means


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--celeba-dir", required=True)
    ap.add_argument("--nonface-dir", required=True)
    ap.add_argument("--output", default="runs/dbvae")
    ap.add_argument("--baseline", action="store_true", help="train CNN instead of DB-VAE")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--kl-weight", type=float, default=5e-4)
    ap.add_argument("--latent-dim", type=int, default=100)
    ap.add_argument("--n-filters", type=int, default=12)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    print(f"device: {device}")

    dataset = CelebAImageNetDataset(args.celeba_dir, args.nonface_dir, image_size=args.image_size)
    print(f"dataset: {len(dataset)} ({len(dataset.face_indices())} face, {len(dataset.nonface_indices())} non-face)")

    if args.baseline:
        model = StandardCNN(image_size=args.image_size, n_filters=args.n_filters).to(device)
    else:
        model = DBVAE(latent_dim=args.latent_dim, image_size=args.image_size, n_filters=args.n_filters).to(device)
    is_dbvae = isinstance(model, DBVAE)

    sampler = BalancedBatchSampler(
        dataset.face_indices(), dataset.nonface_indices(), args.batch_size,
        rng=np.random.default_rng(args.seed),
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=2, pin_memory=(device.type == "cuda"))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        if is_dbvae:
            mu = get_latent_means(model, dataset, dataset.face_indices(), device)
            sampler.set_face_weights(compute_sample_weights(mu))
            print(f"epoch {epoch}: refreshed adaptive weights")

        model.train()
        losses, accs = [], []
        for x, y in tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()

            opt.zero_grad()
            if is_dbvae:
                out = model(x)
                loss, _ = dbvae_loss(x, out.x_recon, y, out.y_logit, out.z_mean, out.z_logvar,
                                     kl_weight=args.kl_weight)
                preds = (torch.sigmoid(out.y_logit).view(-1) > 0.5).float()
            else:
                logits = model(x).view(-1)
                loss = F.binary_cross_entropy_with_logits(logits, y.view(-1))
                preds = (torch.sigmoid(logits) > 0.5).float()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            accs.append((preds == y.view(-1)).float().mean().item())

        print(f"epoch {epoch}: loss={np.mean(losses):.4f}  acc={np.mean(accs):.4f}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "baseline": args.baseline,
        "latent_dim": args.latent_dim,
        "n_filters": args.n_filters,
        "image_size": args.image_size,
    }, out / "model.pt")
    print(f"saved to {out / 'model.pt'}")
