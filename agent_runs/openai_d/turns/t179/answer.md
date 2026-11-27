The onset-2 gate preserves late corrections in the cached states and avoids several early outliers. The last inspection stopped on an array-conversion error without changing weights. I’ll now build the strength-2.6 candidate, verify its exact local loss, and install it only if that reproduces the scan; fresh GPU checks will determine whether it generalizes.

```python
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

torch.set_num_threads(8)

def r157_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

r157_fallback = {k: r157_np(v).astype(np.float32).copy()
                 for k, v in r115_strength2.items()}
r157_candidate = {k: v.copy() for k, v in r157_fallback.items()}
r157_gate_key = (2, 0.1, 3.0)
r157_gate = r152_gates[r157_gate_key]
r157_plus, r157_minus = map(int, r151_slots)

r157_W1 = r157_candidate["attr_heads.value_bin.0.weight"]
r157_b1 = r157_candidate["attr_heads.value_bin.0.bias"]
r157_decoder = (r157_W1[r157_plus].astype(np.float64)
                 - r157_W1[r157_minus].astype(np.float64)) / 2
r157_decoder_b = float((float(r157_b1[r157_plus])
                        - float(r157_b1[r157_minus])) / 2)
r157_W1[r157_plus] = r157_gate["v"] + r157_decoder
r157_W1[r157_minus] = r157_gate["v"] - r157_decoder
r157_b1[r157_plus] = r157_gate["b"] + r157_decoder_b
r157_b1[r157_minus] = r157_gate["b"] - r157_decoder_b
r157_candidate["attr_heads.value_bin.2.weight"][:, [r157_plus, r157_minus]] *= 1.3

r157_H = F.gelu(F.linear(
    r145_X,
    torch.as_tensor(r157_candidate["attr_heads.value_bin.0.weight"]),
    torch.as_tensor(r157_candidate["attr_heads.value_bin.0.bias"])))
r157_L = F.linear(
    r157_H,
    torch.as_tensor(r157_candidate["attr_heads.value_bin.2.weight"]),
    torch.as_tensor(r157_candidate["attr_heads.value_bin.2.bias"])) + r145_transition
r157_loss = r138_nll(r157_L.numpy())
r157_delta = r157_loss - r157_np(r145_loss_orig)
print("pair", r157_plus, r157_minus)
print("local mean NLL delta", float(r157_delta.mean()))
print("largest reading loss", float(r157_delta.max()))
for _event in np.unique(r144_events):
    _mask = r144_events == _event
    print(_event, "n", int(_mask.sum()),
          "delta", float(r157_delta[_mask].mean()))

assert np.isfinite(r157_L.numpy()).all()
assert abs(float(r157_delta.mean()) - 0.002615) < 0.0002, \
       "Exact candidate does not reproduce the gate scan; nothing installed."
for _name, _tensor in r157_candidate.items():
    save_edit(_name, torch.as_tensor(_tensor))
r157_installed = True

with Path("/work/notes.md").open("a") as _f:
    _f.write(
        "\n## Run 157: warm-up insulin trial\n"
        "Installed strength-2.6 insulin with onset-2 gate, C=0.1, sharpness=3. "
        "Paired neurons and decoder unchanged; only their common gate and output "
        "strength changed. Exact cached local NLL delta "
        f"{r157_delta.mean():+.6f}. Strength-2 fallback saved in r157_fallback. "
        "This is an unaccepted trial pending factual and rollout checks.\n")
print("INSTALLED WARM-UP TRIAL; factual validation pending.")
```
