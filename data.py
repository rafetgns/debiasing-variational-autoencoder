from pathlib import Path

import numpy as np
import pandas as pd
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEMOGRAPHIC_KEYS = ("light_female", "light_male", "dark_female", "dark_male")


def default_transform(image_size=64):
    return T.Compose([T.Resize((image_size, image_size), antialias=True), T.ToTensor()])


def _list_images(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"directory not found: {root}")
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not paths:
        raise RuntimeError(f"no images under {root}")
    return paths


class CelebAImageNetDataset(Dataset):
    def __init__(self, celeba_dir, nonface_dir, image_size=64):
        self.face_paths = _list_images(celeba_dir)
        self.nonface_paths = _list_images(nonface_dir)
        self.paths = self.face_paths + self.nonface_paths
        self.labels = np.concatenate([
            np.ones(len(self.face_paths), dtype=np.int64),
            np.zeros(len(self.nonface_paths), dtype=np.int64),
        ])
        self.transform = default_transform(image_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), int(self.labels[idx])

    def face_indices(self):
        return np.arange(len(self.face_paths))

    def nonface_indices(self):
        return np.arange(len(self.face_paths), len(self.paths))


class PPBDataset(Dataset):
    def __init__(self, paths, demographics, image_size=64):
        self.paths = list(paths)
        self.demographics_arr = np.asarray(demographics)
        self.transform = default_transform(image_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), 1, str(self.demographics_arr[idx])

    def demographics(self):
        return self.demographics_arr

    def counts_by_demographic(self):
        d = self.demographics_arr
        return {k: int((d == k).sum()) for k in DEMOGRAPHIC_KEYS}


def load_ppb(ppb_dir, annotations_csv, image_size=64):
    img_dir = Path(ppb_dir) / "images"
    df = pd.read_csv(annotations_csv)
    paths, demos = [], []
    for _, row in df.iterrows():
        p = img_dir / str(row["filename"])
        if not p.exists():
            continue
        skin = "light" if str(row["skin_type"]).lower().startswith("light") else "dark"
        sex = "male" if str(row["gender"]).lower().startswith("m") else "female"
        paths.append(p)
        demos.append(f"{skin}_{sex}")
    if not paths:
        raise FileNotFoundError(f"no PPB images matched under {img_dir}")
    return PPBDataset(paths, demos, image_size=image_size)


def load_ppb_from_folders(root, image_size=64):
    root = Path(root)
    paths, demos = [], []
    for key in DEMOGRAPHIC_KEYS:
        sub = root / key
        if not sub.exists():
            continue
        for p in sorted(sub.rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                paths.append(p)
                demos.append(key)
    if not paths:
        raise FileNotFoundError(f"no fallback eval images under {root}")
    return PPBDataset(paths, demos, image_size=image_size)
