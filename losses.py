import torch
import torch.nn.functional as F


def _reduce(x, reduction):
    if reduction == "none":
        return x
    if reduction == "sum":
        return x.sum()
    return x.mean()


def kl_divergence(mean, logvar, reduction="mean"):
    per = 0.5 * torch.sum(torch.exp(logvar) + mean.pow(2) - 1.0 - logvar, dim=1)
    return _reduce(per, reduction)


def reconstruction_l1(x, x_recon, reduction="mean"):
    per = torch.mean(torch.abs(x - x_recon), dim=list(range(1, x.dim())))
    return _reduce(per, reduction)


def dbvae_loss(x, x_recon, y, y_logit, mean, logvar, kl_weight=5e-4):
    y = y.float().view(-1)
    y_logit = y_logit.view(-1)
    cls = F.binary_cross_entropy_with_logits(y_logit, y, reduction="none")
    kl = kl_divergence(mean, logvar, reduction="none")
    rec = reconstruction_l1(x, x_recon, reduction="none")
    face = (y > 0.5).float()
    total = cls + face * (kl_weight * kl + rec)
    return total.mean(), cls.mean()
