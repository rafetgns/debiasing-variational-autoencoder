import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import DEMOGRAPHIC_KEYS, load_ppb, load_ppb_from_folders
from models import DBVAE, StandardCNN


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    return imgs, [b[1] for b in batch], [b[2] for b in batch]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ppb-dir")
    ap.add_argument("--ppb-csv")
    ap.add_argument("--test-faces-dir")
    args = ap.parse_args()

    device = pick_device()
    ckpt = torch.load(args.model, map_location=device)
    if ckpt.get("baseline"):
        model = StandardCNN(image_size=ckpt["image_size"], n_filters=ckpt["n_filters"]).to(device)
    else:
        model = DBVAE(latent_dim=ckpt["latent_dim"], image_size=ckpt["image_size"], n_filters=ckpt["n_filters"]).to(device)
    model.load_state_dict(ckpt["state_dict"])

    if args.ppb_dir and args.ppb_csv:
        ppb = load_ppb(args.ppb_dir, args.ppb_csv, image_size=ckpt["image_size"])
    elif args.test_faces_dir:
        ppb = load_ppb_from_folders(args.test_faces_dir, image_size=ckpt["image_size"])
    else:
        ap.error("provide --ppb-dir/--ppb-csv or --test-faces-dir")

    loader = DataLoader(ppb, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate)
    probs, labels, demos = [], [], []
    with torch.inference_mode():
        for imgs, ys, gs in loader:
            probs.append(torch.sigmoid(model.predict(imgs.to(device))).view(-1).cpu().numpy())
            labels.append(np.asarray(ys, dtype=np.int64))
            demos.append(np.asarray(gs))
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    demos = np.concatenate(demos)
    preds = (probs >= 0.5).astype(np.int64)


    print(f"\nface detection rate (all groups): {(labels == preds).mean():.4f}  (n={len(ppb)})")
    print("per-demographic detection rate:")
    per = {}
    for k in DEMOGRAPHIC_KEYS:
        m = demos == k
        if not m.any():
            continue
        per[k] = (labels[m] == preds[m]).mean()
        print(f"  {k:<14s}  rate={per[k]:.4f}  mean p(face)={probs[m].mean():.4f}  n={int(m.sum())}")
    if len(per) >= 2:
        print(f"max-min gap: {max(per.values()) - min(per.values()):.4f}")
