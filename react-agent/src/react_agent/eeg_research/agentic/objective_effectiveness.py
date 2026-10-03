"""Gradient probes use same-image IDs with genuinely tied frozen targets."""
from __future__ import annotations
from typing import Any

def gradient_probe(objective: Any) -> dict[str, Any]:
    import torch
    from react_agent.eeg_training.model import contrastive_loss
    from react_agent.eeg_training.hooks import compute_objective,objective_parameters
    rows=[];parameters=objective_parameters(objective)
    with torch.random.fork_rng(devices=[]):
        for seed in (1729,2718,3141):
            torch.manual_seed(seed)
            codes=torch.tensor([0,0,1,2,3,3,4,5]);ids=[str(int(k)) for k in codes]
            image=torch.randn(6,1024)[codes]
            eeg=torch.randn(8,1024,requires_grad=True)
            scale=torch.tensor(2.75,requires_grad=True)
            baseline=contrastive_loss(eeg,image,scale)
            candidate=compute_objective(objective,eeg,image,scale,ids)
            a=torch.autograd.grad(baseline,(eeg,scale),retain_graph=True)
            b=torch.autograd.grad(candidate,(eeg,scale,*parameters),allow_unused=True)
            same_eeg=b[0] is not None and torch.allclose(a[0],b[0],atol=2e-6,rtol=2e-5)
            same_scale=b[1] is not None and torch.allclose(a[1],b[1],atol=2e-6,rtol=2e-5)
            own=max([float(v.detach().abs().max()) for v in b[2:] if v is not None]+[0.0])
            finite=bool(torch.isfinite(candidate)) and all(v is None or bool(torch.isfinite(v).all()) for v in b)
            if not finite:raise ValueError('non_finite_objective_gradient_probe')
            rows.append({'seed':seed,'loss_abs_difference':float((candidate-baseline).detach().abs()),
                         'eeg_gradient_max_abs_difference':None if b[0] is None else float((b[0]-a[0]).detach().abs().max()),
                         'scale_gradient_abs_difference':None if b[1] is None else float((b[1]-a[1]).detach().abs()),
                         'learnable_objective_gradient_max':own,
                         'training_gradient_changed':bool(not same_eeg or not same_scale or own>2e-6)})
    changed=any(r['training_gradient_changed'] for r in rows)
    return {'status':'gradient_changed_on_tied_target_probes' if changed else 'baseline_equivalent_on_tied_target_probes',
            'gradient_changed':changed,'probes':rows,'scope':'Three synthetic CPU probes with identical image vectors for identical IDs; no training score. Negative probes require source-level interpretation; positive probes do not establish benefit.'}


def objective_switch(config: dict[str, Any]) -> str | None:
    """Resolve the declared switch without rewriting an approved experiment."""
    explicit = config.get("ablation_switch")
    if isinstance(explicit, str) and explicit:
        return explicit
    if isinstance(config.get("objective_variant_on"), bool):
        return "objective_variant_on"
    return None


def objective_is_active(config: dict[str, Any]) -> bool:
    switch = objective_switch(config)
    return config.get(switch, config.get("ablation_default", True)) is True if switch else True
