"""EEGProjectLayer and the symmetric contrastive loss from the baseline trainer."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ResidualAdd(nn.Module):
    """Add a residual branch."""

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the input plus the branch."""
        return inputs + self.block(inputs)


class EEGProjectLayer(nn.Module):
    """Flatten the selected window and project it to the image feature size."""

    def __init__(self, z_dim: int, c_num: int, timesteps: list[int], drop_proj: float = 0.3) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps
        self.input_dim = c_num * (timesteps[1] - timesteps[0])
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, z_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(z_dim, z_dim),
                    nn.Dropout(drop_proj),
                )
            ),
            nn.LayerNorm(z_dim),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project one batch of [batch, channels, time]."""
        flat = inputs.reshape(inputs.shape[0], self.input_dim)
        return self.model(flat)


class LocalRetrieval(nn.Module):
    """Contrastive step on one replica. DataParallel keeps each card's own batch."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        if not hasattr(encoder, "logit_scale"):
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.softplus = nn.Softplus()

    def forward(self, eeg: torch.Tensor, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the local loss and within-batch top-1 and top-5."""
        embedding = self.encoder(eeg)
        if hasattr(self.encoder, "logit_scale"):
            scale = self.encoder.softplus(self.encoder.logit_scale)
        else:
            scale = self.softplus(self.logit_scale)
        loss = contrastive_loss(embedding, image, scale)
        eeg_n = embedding / embedding.norm(dim=-1, keepdim=True)
        image_n = image / image.norm(dim=-1, keepdim=True)
        similarity = eeg_n @ image_n.T
        labels = torch.arange(similarity.shape[0], device=similarity.device)
        top1 = (similarity.argmax(dim=-1) == labels).float().mean()
        picked = similarity.topk(min(5, similarity.shape[1]), dim=-1).indices
        top5 = (picked == labels[:, None]).any(dim=-1).float().mean()
        return loss, top1, top5


def contrastive_loss(eeg_z: torch.Tensor, img_z: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE. Image embeddings are normalized; EEG scale stays in logit_scale."""
    img_z = img_z / img_z.norm(dim=-1, keepdim=True)
    logits = logit_scale * eeg_z @ img_z.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    image_loss = F.cross_entropy(logits, labels)
    text_loss = F.cross_entropy(logits.T, labels)
    return (image_loss + text_loss) / 2


def within_batch_accuracy(eeg_z: torch.Tensor, img_z: torch.Tensor) -> tuple[float, float]:
    """Top-1 and top-5 retrieval inside one batch, matching the source epoch metric."""
    eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)
    img_z = img_z / img_z.norm(dim=-1, keepdim=True)
    similarity = eeg_z @ img_z.T
    labels = torch.arange(similarity.shape[0], device=similarity.device)
    top1 = similarity.argmax(dim=-1)
    top5 = similarity.topk(min(5, similarity.shape[1]), dim=-1).indices
    top1_acc = (top1 == labels).float().mean().item()
    top5_acc = (top5 == labels[:, None]).any(dim=-1).float().mean().item()
    return top1_acc, top5_acc
