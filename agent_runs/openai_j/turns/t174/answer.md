The fit is a scalar tilt of the value-bin logits, applied to the raw head input. I’ll encode a moderately amplified glucose correction using two replaceable head units and audit it without saving it yet. Exact logit agreement and tuning NLL degradation below 0.005 would justify a GPU test; either failure would reject it.

```python
import numpy as np
import torch
import torch.nn.functional as _F158
from pathlib import Path

trial158_gain = 1.4
trial158_coef = np.asarray(wm_fits152[(1.0, 0.0001)]) * trial158_gain
trial158_free = np.array([170, 89])
trial158_keep = np.setdiff1d(np.arange(192), trial158_free)
trial158_W0 = wm_weights145["attr_heads.value_bin.0.weight"].clone()
trial158_b0 = wm_weights145["attr_heads.value_bin.0.bias"].clone()
trial158_W2 = wm_weights145["attr_heads.value_bin.2.weight"].clone()
trial158_b2 = wm_weights145["attr_heads.value_bin.2.bias"].clone()

# Algebraic compression of the two old units, fitted on factual prior draws.
trial158_H = wm_train145["hidden"].reshape(-1, 192).double()
trial158_K = torch.cat([
    trial158_H[:, trial158_keep],
    torch.ones((len(trial158_H), 1), dtype=torch.float64)
], dim=1)
trial158_Y = (
    trial158_H[:, trial158_free]
    @ trial158_W2[:, trial158_free].double().T
)
trial158_cov = trial158_K.T @ trial158_K / len(trial158_K)
trial158_rhs = trial158_K.T @ trial158_Y / len(trial158_K)
trial158_comp = torch.linalg.solve(
    trial158_cov + 1e-9 * torch.eye(len(trial158_keep) + 1, dtype=torch.float64),
    trial158_rhs
)
trial158_W2[:, trial158_keep] += trial158_comp[:-1].T.float()
trial158_b2 += trial158_comp[-1].float()

# GELU(a+t)-GELU(a) is t in the glucose-positive gate and zero in MAP.
trial158_gate_strength = 200.0
trial158_gate = torch.as_tensor(wm_gate153, dtype=torch.float32)
trial158_c = torch.as_tensor(trial158_coef, dtype=torch.float32)
trial158_W0[trial158_free[0]] = (
    trial158_gate_strength * trial158_gate[:-1] + trial158_c[:-1]
)
trial158_b0[trial158_free[0]] = (
    trial158_gate_strength * trial158_gate[-1] + trial158_c[-1]
)
trial158_W0[trial158_free[1]] = trial158_gate_strength * trial158_gate[:-1]
trial158_b0[trial158_free[1]] = trial158_gate_strength * trial158_gate[-1]
trial158_W2[:, trial158_free[0]] = wm_v151
trial158_W2[:, trial158_free[1]] = -wm_v151
trial158_weights = {
    "attr_heads.value_bin.0.weight": trial158_W0,
    "attr_heads.value_bin.0.bias": trial158_b0,
    "attr_heads.value_bin.2.weight": trial158_W2,
    "attr_heads.value_bin.2.bias": trial158_b2,
}

def trial158_audit(mc, glucose_mask):
    x = mc["inputs"]
    new_h = _F158.gelu(_F158.linear(x, trial158_W0, trial158_b0))
    new_y = _F158.linear(new_h, trial158_W2, trial158_b2)
    old_y = _F158.linear(
        mc["hidden"],
        wm_weights145["attr_heads.value_bin.2.weight"],
        wm_weights145["attr_heads.value_bin.2.bias"]
    )
    logits = mc["logits"] + new_y - old_y
    p = torch.softmax(logits, -1)
    idx = mc["target"][None, :, None].expand(p.shape[0], -1, 1)
    nll = -p.gather(-1, idx).squeeze(-1).reshape(
        -1, 4, p.shape[1]
    ).mean(1).clamp_min(1e-30).log()
    mean = (p.mean(0) * mc["centres"]).sum(-1)
    expected = wm_rawloss152(mc, trial158_coef, glucose_mask)
    expected_delta = expected[2][..., None] * wm_v151
    err = new_y - old_y - expected_delta
    err = err - err.mean(-1, keepdim=True)
    gate_values = x @ trial158_gate[:-1] + trial158_gate[-1]
    return nll, mean, {
        "logit_equivalence_rms": float(err.square().mean().sqrt()),
        "glucose_gate_min": float(gate_values[:, glucose_mask].min()),
        "MAP_gate_max": float(gate_values[:, ~glucose_mask].max()),
        "max_abs_tilt": float(expected[2].abs().max()),
        "dNLL": float((nll - mc["nll"]).mean()),
        "ideal_tilt_dNLL": float((expected[0] - mc["nll"]).mean()),
    }

trial158_results = {}
for trial158_name, trial158_mc, trial158_mask in (
    ("train", wm_train145, wm_glucose152),
    ("tuning", wm_tune145, wm_tune_glucose152),
):
    trial158_nll, trial158_mean, trial158_info = trial158_audit(
        trial158_mc, trial158_mask
    )
    trial158_results[trial158_name] = trial158_info
    print(trial158_name, trial158_info)
    if trial158_name == "tuning":
        for trial158_event in sorted(wm_tune_meta145.event.unique()):
            trial158_sel = wm_tune_meta145.event.to_numpy() == trial158_event
            print(" ", trial158_event, "dNLL",
                  float((trial158_nll - trial158_mc["nll"])[:, trial158_sel].mean()))
print("GPU validation stay IDs:", list(repair_validation_ids))
print("Active edits:", edited())
with Path("/work/notes.md").open("a") as trial158_f:
    trial158_f.write(
        "\n## Run 158: stochastic glucose-head candidate audit\n"
        "- Built, but did not save, gain-1.4 raw-input correction from run 152.\n"
        "- Replaced units 170 and 89 after algebraic compression using factual prior draws.\n"
        "- Strong linear sensor gate implements glucose-only correction.\n"
        f"- Local audits: {trial158_results}\n"
        "- GPU factual and rollout validation are still required; no active edits.\n"
    )
```
