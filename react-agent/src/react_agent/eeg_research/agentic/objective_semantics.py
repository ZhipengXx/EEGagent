"""Prospective explicit probability-loss contract and descriptive CPU probes."""
from __future__ import annotations
from typing import Any

EXPLICIT_VARIANT = "positive_mass_symmetric_infonce"

def reference_loss(eeg: Any, image: Any, scale: Any, ids: Any) -> Any:
    import torch
    image = image / image.norm(dim=-1, keepdim=True)
    logits = scale * eeg @ image.T
    positive = torch.tensor([[a == b for b in ids] for a in ids], device=eeg.device)
    terms = []
    for scores, mask in ((logits, positive), (logits.T, positive.T)):
        terms.append((torch.logsumexp(scores, 1) -
                      torch.logsumexp(scores.masked_fill(~mask, float("-inf")), 1)).mean())
    return sum(terms) / 2


def semantic_probe(objective: Any) -> dict[str, Any]:
    import torch
    from react_agent.eeg_training.hooks import compute_objective
    cases = {"unique_ids": list(range(8)), "duplicate_ids": [0,0,1,2,3,3,4,5],
             "all_same_id": [0]*8}
    rows = []
    with torch.random.fork_rng(devices=[]):
        for seed in (1729, 2718):
            for name, labels in cases.items():
                torch.manual_seed(seed)
                ids = [str(i) for i in labels]
                codes = torch.tensor(labels)
                image = torch.randn(max(labels)+1, 1024, dtype=torch.float64)[codes]
                eeg = torch.randn(8,1024,dtype=torch.float64,requires_grad=True)
                scale = torch.tensor(2.75,dtype=torch.float64,requires_grad=True)
                expected = reference_loss(eeg,image,scale,ids)
                expected_grad = torch.autograd.grad(expected,(eeg,scale),retain_graph=True)
                row = {"seed":seed,"case":name,"reference_loss":float(expected.detach())}
                try:
                    actual = compute_objective(objective,eeg,image,scale,ids)
                    grads = torch.autograd.grad(actual,(eeg,scale),allow_unused=True)
                    finite = bool(torch.isfinite(actual)) and all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
                    row.update(actual_loss=float(actual.detach()),finite=finite,
                               loss_matches=bool(torch.allclose(actual,expected,atol=1e-7,rtol=1e-6)),
                               gradients_match=all(g is not None and bool(torch.allclose(g,h,atol=1e-7,rtol=1e-6)) for g,h in zip(grads,expected_grad)))
                except Exception as exc:
                    row.update(finite=False,loss_matches=False,gradients_match=False,error_type=type(exc).__name__,error=str(exc)[:500])
                rows.append(row)
    matches = all(r["finite"] and r["loss_matches"] and r["gradients_match"] for r in rows)
    return {"reference_definition":EXPLICIT_VARIANT,"matches_reference":matches,
            "unique_ids_reduce_to_baseline":all(r["loss_matches"] and r["gradients_match"] for r in rows if r["case"]=="unique_ids"),
            "all_same_id_finite_zero_loss":all(r["finite"] and abs(r.get("actual_loss",float("inf")))<1e-7 for r in rows if r["case"]=="all_same_id"),
            "probes":rows,"scope":"Synthetic CPU semantic probes; raw EEG logits, normalized frozen images, all same-ID positives including diagonal, all local items in denominator, symmetric mean. These define only the explicit prospective variant. Historical unnamed/ambiguous variants retain their scores and are described rather than silently redefined."}
