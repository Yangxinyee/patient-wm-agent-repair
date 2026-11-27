The feature code confirms the 16-versus-17-centre error. Earlier analyses support a negative insulin effect, but norepinephrine estimates depend on the adjustment assumptions. I’ll fix the feature extraction and measure its agreement with GPU predictions. Poor agreement would limit how much I can trust it when designing an edit.

```python
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import re

def state_matrix(obj):
    if isinstance(obj, np.ndarray):
        return obj
    keys = obj.files if hasattr(obj, "files") else obj.keys()
    candidates = [(k, obj[k]) for k in keys
                  if getattr(obj[k], "ndim", 0) == 2 and obj[k].shape[1] == 992]
    if len(candidates) != 1:
        raise ValueError([(k, getattr(obj[k], "shape", None)) for k in keys])
    return candidates[0][1]

_repair_bin_score = torch.tensor(
    [0.0] + list(np.arange(1, 17, dtype=float) - 8.5), dtype=torch.float32)

def repair_head_features(meta, states):
    parts = cpu_head_parts(meta, states)
    last_z = torch.tensor(meta.last_z.to_numpy(), dtype=torch.float32)
    additive = parts["additive"] + last_z[:, None] * (
        W["last_numeric.weight"][:, 6] - W["last_numeric.weight"][:, 2])[None, :]
    q = ln(parts["base"] + lin(torch.zeros(len(meta), 32), "z_head")
           + W["type_emb.weight"][1], "query_norm")
    hidden = F.gelu(lin(q + additive, "attr_heads.value_bin.0"))
    logits = lin(hidden, "attr_heads.value_bin.2") + parts["transition"]
    probs = logits.softmax(-1)
    centres = torch.tensor(np.stack([
        [0.0] + list(VOCAB["value_bin_centres"][event]) for event in meta.event
    ]), dtype=torch.float32)
    means = (probs * centres).sum(-1)
    score_mean = (probs * _repair_bin_score).sum(-1)
    covariance = (probs * centres * _repair_bin_score).sum(-1) - means * score_mean
    observed = torch.tensor(meta.observed_bin.to_numpy(), dtype=torch.long)
    nll = -probs.gather(1, observed[:, None]).squeeze(1).clamp_min(1e-12).log()
    return {"hidden": hidden.numpy(), "probs": probs.numpy(),
            "mean": means.numpy(), "covariance": covariance.numpy(),
            "nll": nll.numpy()}

repair_train = repair_head_features(head_meta_train, state_matrix(state_files[4]))
repair_tune = repair_head_features(head_meta_tune, state_matrix(state_files[3]))
print("Feature shapes:", repair_train["hidden"].shape, repair_tune["hidden"].shape)
for event in ("glucose_finger", "map_arterial"):
    mask = head_meta_tune.event.to_numpy() == event
    print(event, "CPU/GPU mean RMSE",
          float(np.sqrt(np.mean((repair_tune["mean"][mask] - head_gpu_mean[mask])**2))),
          "CPU NLL", float(repair_tune["nll"][mask].mean()),
          "GPU NLL", float(head_gpu_nll[mask].mean()))

print("\n=== TWO EARLIER NE DIAGNOSTIC SECTIONS ===")
sections = re.split(r"(?m)^## ", Path("/work/notes.md").read_text())
for section in sections:
    if section.startswith(("Factual baseline and dynamic", "More efficient dynamic")):
        print("\n##", section[:1800])

with open("/work/notes.md", "a") as f:
    f.write("\n## Recovery; Python 49, forward 4\n"
            "- Fixed feature extraction's absent-bin centre; cached training/tuning hidden features.\n"
            "- Reconstruction remains approximate; actual GPU factual tests will decide edit acceptance.\n"
            "- Insulin threshold evidence is stronger than norepinephrine causal identification.\n")
print("Edits:", edited())
```
