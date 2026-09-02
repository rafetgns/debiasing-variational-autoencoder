import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import Sampler


class BalancedBatchSampler(Sampler):

    def __init__(self, face_indices, nonface_indices, batch_size, epoch_length=None, face_weights=None, rng=None):
        if batch_size < 2 or batch_size % 2 != 0:
            raise ValueError(f"batch_size must be even, got {batch_size}")
        self.face_indices = np.asarray(face_indices, dtype=np.int64)
        self.nonface_indices = np.asarray(nonface_indices, dtype=np.int64)
        self.batch_size = batch_size
        self.half = batch_size // 2
        self.epoch_length = epoch_length or max(len(face_indices) // self.half, 1)
        self.rng = rng or np.random.default_rng()
        self.face_weights = face_weights

    def set_face_weights(self, weights):
        w = np.asarray(weights, dtype=np.float64)
        self.face_weights = w / w.sum()

    def __iter__(self):
        for _ in range(self.epoch_length):
            face = self.rng.choice(self.face_indices, size=self.half, replace=True, p=self.face_weights)
            non = self.rng.choice(self.nonface_indices, size=self.half, replace=True)
            batch = np.concatenate([face, non])
            self.rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self.epoch_length


@torch.inference_mode()
def get_latent_means(model, dataset, face_indices, device, batch_size=128):
    model.eval()
    loader = DataLoader(Subset(dataset, list(face_indices.tolist())), batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = []
    for batch in loader:
        chunks.append(model.latent_mean(batch[0].to(device)).cpu())
    return torch.cat(chunks, dim=0).numpy()


def compute_sample_weights(latent_means, bins=10, alpha=0.001):
    n, d = latent_means.shape
    log_w = np.zeros(n, dtype=np.float64)
    for j in range(d):
        values = latent_means[:, j].astype(np.float64)
        density, edges = np.histogram(values, bins=bins, density=True)
        edges = edges.copy()
        edges[0], edges[-1] = -np.inf, np.inf
        idx = np.clip(np.digitize(values, edges) - 1, 0, bins - 1)
        smoothed = density + alpha
        smoothed = smoothed / smoothed.sum()
        p_j = 1.0 / smoothed[idx]
        p_j = p_j / p_j.sum()
        log_w += np.log(p_j + 1e-12)
    log_w -= log_w.max()  # stability before exp
    w = np.exp(log_w)
    return w / w.sum()
