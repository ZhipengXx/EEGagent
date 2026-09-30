"""CPU sentinel tests for training hooks. No GPU and no live Goal."""

from __future__ import annotations

import torch
from torch import nn

from react_agent.eeg_training.hooks import (
    apply_train_transform,
    compute_objective,
    image_id_duplicate_stats,
    is_custom_objective,
    note_eval_without_transform,
    objective_parameters,
)
from react_agent.eeg_training.model import EEGProjectLayer, contrastive_loss


def test_image_id_duplicate_rate_does_not_need_encoder() -> None:
    stats = image_id_duplicate_stats(["a", "a", "b", "c"])
    assert stats["batch_size"] == 4
    assert stats["unique_image_ids"] == 3
    assert stats["duplicate_rate"] == 0.25
    assert stats["max_same_image"] == 2
    assert stats["same_image_positives"] == 1


def test_custom_objective_changes_gradients() -> None:
    eeg = torch.randn(4, 8, requires_grad=True)
    img = torch.randn(4, 8)
    scale = torch.tensor(1.0)

    base = contrastive_loss(eeg, img, scale)
    base.backward()
    first = eeg.grad.detach().clone()
    eeg.grad = None

    def shifted(eeg_z, img_z, logit_scale, positives=None):
        return contrastive_loss(eeg_z, img_z, logit_scale) + eeg_z.mean() * 0.5

    assert is_custom_objective(shifted)
    loss = compute_objective(shifted, eeg, img, scale)
    loss.backward()
    assert not torch.allclose(eeg.grad, first)


def test_transform_is_train_only() -> None:
    counters = {"transform_train_calls": 0, "transform_eval_calls": 0}

    def noise(eeg):
        return eeg + 1

    out = apply_train_transform(noise, torch.zeros(2, 3), counters)
    assert torch.equal(out, torch.ones(2, 3))
    assert counters["transform_train_calls"] == 1
    note_eval_without_transform(counters)
    assert counters["transform_eval_calls"] == 0


def test_optimizer_contains_objective_parameters() -> None:
    class Weighted(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.ones(1))

        def forward(self, eeg_z, img_z, scale, positives=None):
            return contrastive_loss(eeg_z, img_z, scale) * self.w.abs()

    encoder = EEGProjectLayer(8, 2, [0, 4])
    objective = Weighted()
    params = list(encoder.parameters()) + objective_parameters(objective)
    opt = torch.optim.AdamW(params, lr=1e-3)
    owned = [item for group in opt.param_groups for item in group["params"]]
    assert any(item is objective.w for item in owned)


def test_getitem_keeps_image_id_out_of_encoder_keys() -> None:
    from react_agent.eeg_training.data import RetrievalTrials, collate_retrieval

    eeg = torch.zeros(17, 250)
    features = torch.zeros(1024)
    dataset = RetrievalTrials(
        [{"eeg": eeg, "img": "img_a", "img_features": features}],
        [0, 250],
    )
    row = dataset[0]
    assert {"eeg", "img_features", "image_id"} <= set(row)
    assert row["image_id"] == "img_a"
    assert "query_id" in row
    assert "subject" in row
    assert row["eeg"].shape == (17, 250)
    batched = collate_retrieval([row, row])
    assert batched["image_id"] == ["img_a", "img_a"]
    assert "query_id" in batched
    assert batched["eeg"].shape[1:] == (17, 250)
    encoder = EEGProjectLayer(1024, 17, [0, 250])
    encoded = encoder(batched["eeg"])
    assert encoded.shape[0] == 2
