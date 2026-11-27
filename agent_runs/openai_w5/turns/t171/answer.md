The corrected sensor-specific fits now have the expected signs. Full insulin strength with half norepinephrine strength adds about 0.00136 nats on the calibration holdout, leaving room under the factual-loss limit. I’ll install that reversible candidate, then test it on the GPU; excessive factual loss or weak rollout responses would reject it.

```python
import torch.nn.functional as F
from scipy.special import softmax

_r149_strengths = (1.0, 1.0, .5, .5)
_r149_names = [
    "attr_heads.value_bin.0.weight",
    "attr_heads.value_bin.0.bias",
    "attr_heads.value_bin.2.weight",
    "attr_heads.value_bin.2.bias",
]
_r149_candidate = {n: _r147_clean_weights[n].clone() for n in _r149_names}
_r149_W1 = _r149_candidate[_r149_names[0]]
_r149_b1 = _r149_candidate[_r149_names[1]]
_r149_W2 = _r149_candidate[_r149_names[2]]
_r149_b2 = _r149_candidate[_r149_names[3]]

_r149_W2[:, _r148_keep] += torch.as_tensor(
    _r148_distill.coef_, dtype=torch.float32)
_r149_W2[:, _r148_reserved] = 0
_r149_b2 += torch.as_tensor(_r148_distill.intercept_, dtype=torch.float32)
for k, (spec, strength) in enumerate(zip(_r148_specs, _r149_strengths)):
    signal_neuron, control_neuron = _r148_reserved[2*k:2*k+2]
    _r149_W1[signal_neuron] = torch.as_tensor(spec["row_signal"], dtype=torch.float32)
    _r149_b1[signal_neuron] = float(spec["bias_signal"])
    _r149_W1[control_neuron] = torch.as_tensor(spec["row_control"], dtype=torch.float32)
    _r149_b1[control_neuron] = float(spec["bias_control"])
    _r149_W2[:, signal_neuron] = torch.as_tensor(
        strength*spec["out"], dtype=torch.float32)
    _r149_W2[:, control_neuron] = torch.as_tensor(
        -strength*spec["out"], dtype=torch.float32)

assert all(torch.isfinite(v).all() for v in _r149_candidate.values())
_r149_wc32 = torch.as_tensor(_r147_wc, dtype=torch.float32)
_r149_transition = (
    np.asarray(_r140_features["logits"], dtype=np.float64)
    - _r148_H @ _r141_W2.T - _r141_b2
)
_r149_H32 = F.gelu(F.linear(_r149_wc32, _r149_W1, _r149_b1))
_r149_logits32 = F.linear(_r149_H32, _r149_W2, _r149_b2).numpy() + _r149_transition
_r149_expected = _r148_base + _r148_I + .5*_r148_NE
_r149_P32 = softmax(_r149_logits32, axis=-1).mean(axis=0)
_r149_NLL32 = -np.log(
    _r149_P32[np.arange(len(_r149_P32)), _r141_bins].clip(1e-12))
_r149_local_delta = np.mean(
    (_r149_NLL32-_r141_baseNLL)[_r147_test])
print("Float32 versus analytic logit RMS:",
      np.sqrt(np.mean((_r149_logits32-_r149_expected)**2)))
print("Float32 held-out incremental NLL:", _r149_local_delta)
assert _r149_local_delta < .0031, "Offline factual budget failed; edits not installed."
for n in _r149_names:
    save_edit(n, _r149_candidate[n])
print("Installed sensor-specific causal head candidate.")
print("Tuning-A stay IDs:", list(target_ids_A))
print("Record-derived tuning-A targets:",
      {str(k): round(v[0], 4) for k, v in target_checks_A.items()})

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Runs 147–149: sensor-specific causal-head candidate\n"
        "- Run 147's zero insulin tilt was a constrained-fit sign bug, corrected in run 148; it is not evidence of an absent insulin effect.\n"
        "- Freed eight low-energy value-head neurons by frozen-feature ridge distillation; no model gradients.\n"
        "- Separate, clean sensor gates; smooth monotone logit tilts calibrated to record-derived one-step effects.\n"
        "- Centered decoded exposure using independent-stay-validated ordinary-feature regressions.\n"
        "- Installed insulin strength 1 and NE strength .5 to reserve factual-loss budget.\n"
        f"- Float32 local holdout NLL increment {float(_r149_local_delta):.6f} above clean transport; GPU factual and finite-dose tests still required.\n"
    )
```
