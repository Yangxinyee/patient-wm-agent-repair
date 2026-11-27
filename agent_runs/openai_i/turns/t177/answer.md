Prior sampling explains most of the MAP discrepancy, but glucose still has a bias. The captured register places the previous reading in a suspicious numeric position. I’ll test the possible layouts against the saved predictions. A valid layout should reduce glucose error to roughly the prior-sampling level without worsening MAP.

```python
import numpy as np
import torch
from pathlib import Path

numeric_audit160 = {}
for event160, items160 in validation_samples158.items():
    predictions160 = {p: [] for p in (2, 4, 6)}
    targets160, noise_sd160 = [], []
    for key160, i160 in items160:
        raw160, cap160 = capture159(
            torch.tensor(S[i160:i160+1, 192:960]).reshape(1, 4, 192),
            torch.tensor(S[i160:i160+1, :192]),
            stays156[i160], hours156[i160], event160)
        targets160.append(lookup158[key160])
        prior160 = mlp136(cap160["ctx"], "z_prior")
        draws160 = prior160[:, :32] + noise159 * torch.exp(prior160[:, 32:].clamp(-7, 2))
        q160 = norm136(
            cap160["ctx"] + linear136(draws160, "z_head")
            + W["type_emb.weight"][1], "query_norm")
        transition160 = W["embed_transition.value_bin.weight"][
            codes[event160] * 17 + int(cap160["last_bin"])
        ]
        for position160 in (2, 4, 6):
            reg160 = cap160["register"].clone()
            value160 = reg160[:, 6].clone()
            reg160[:, 6] = 0
            reg160[:, position160] = value160
            wc160 = (
                cap160["with_code"] + q160 - cap160["q"]
                + linear136(reg160, "last_numeric")
                - linear136(cap160["register"], "last_numeric")
            )
            logits160 = mlp136(wc160, "attr_heads.value_bin") + transition160
            means160 = torch.softmax(logits160, -1) @ centres153(event160)
            predictions160[position160].append(float(means160.mean()))
            if position160 == 2:
                noise_sd160.append(float(means160.std()))
    targets160 = np.array(targets160)
    print("\nNUMERIC REGISTER AUDIT:", event160)
    for position160, pred160 in predictions160.items():
        errors160 = np.array(pred160) - targets160
        print("value position", position160,
              "RMSE", round(float(np.sqrt(np.mean(errors160**2))), 5),
              "bias", round(float(np.mean(errors160)), 5))
    print("Four-draw sampling RMSE, position 2:",
          round(float(np.sqrt(np.mean(np.array(noise_sd160)**2) / 4)), 5))
    numeric_audit160[event160] = (predictions160, targets160, noise_sd160)

print("\n=== EARLIER HEAD-RECONSTRUCTION NOTES ===")
print(Path("/work/notes.md").read_text()[10500:15100])

print("\n=== ACTION / RECURRENT CACHE CONTRACTS ===")
for name160 in sorted(list(globals())):
    if name160.startswith("_"):
        continue
    obj160 = globals()[name160]
    if (name160.endswith(("152", "153", "136"))
        and isinstance(obj160, (np.ndarray, torch.Tensor))):
        print(name160, tuple(obj160.shape))
    elif name160.endswith("152") and isinstance(obj160, dict):
        print(name160, "dict keys:", list(obj160)[:12])
for name160 in ("attention136", "block136", "norm136", "core_step136"):
    fn160 = globals()[name160]
    print(name160, "constants:", repr(fn160.__code__.co_consts)[:600])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 160 — previous-value numeric position audit\n"
            "- Prior sampling explains MAP mismatch but leaves a glucose bias.\n"
            "- Testing the previous reading in numeric positions 2, 4 and 6 against saved factual predictions.\n"
            "- No weights changed; reconstructing action and recurrent cache contracts for finite differences.\n")
```