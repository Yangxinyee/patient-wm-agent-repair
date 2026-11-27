The original rollouts show only small dose responses, while the record-based estimates suggest stronger effects. Reconstruction errors are also much larger for glucose rows with previous readings. I’ll test the possible register-flag assignments without changing weights. A substantially lower error and an NLL near the forward tool’s result would identify a usable layout.

```python
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

torch.set_num_threads(8)
probe21 = head_rows.sample(650, random_state=2101).copy()
capture21 = {}
original_linear21 = linear_named

def capture_linear21(*args, **kwargs):
    x, name = args[:2]
    if name in ("last_numeric", "attr_heads.value_bin.0"):
        capture21[name] = x.detach().clone()
    return original_linear21(*args, **kwargs)

linear_named = capture_linear21
try:
    baseline21, probs21, hidden21 = head2(probe21, z_samples=0)
finally:
    linear_named = original_linear21

lf21 = capture21["last_numeric"]
wc21 = capture21["attr_heads.value_bin.0"]
w121 = head_weights["attr_heads.value_bin.0.weight"]
b121 = head_weights["attr_heads.value_bin.0.bias"]
w221 = head_weights["attr_heads.value_bin.2.weight"]
b221 = head_weights["attr_heads.value_bin.2.bias"]
effect21 = head_weights["last_numeric.weight"].T @ w121.T
prebase21 = wc21 @ w121.T + b121 - lf21 @ effect21
has21 = torch.tensor(probe21.last_value.notna().to_numpy(), dtype=torch.float32)
z21 = torch.tensor(probe21.last_z.to_numpy(), dtype=torch.float32)
age21 = np.maximum(
    probe21.hour.to_numpy() -
    probe21.last_hour.fillna(probe21.hour).to_numpy(), 0
)
cid21 = torch.tensor([code_ids[e] for e in probe21.event], dtype=torch.long)
bin21 = torch.tensor(probe21.last_bin.to_numpy(), dtype=torch.long)
transition21 = head_weights["embed_transition.value_bin.weight"][cid21 * 17 + bin21]
centres21 = torch.tensor(
    [[0.0] + list(VOCAB["value_bin_centres"][e]) for e in probe21.event],
    dtype=torch.float32
)
observed_bins21 = torch.tensor([
    1 + np.searchsorted(numeric_specs[(e, "value")]["fine_bin_edges"], v, side="left")
    for e, v in zip(probe21.event, probe21.observed)
], dtype=torch.long)
reference21 = torch.tensor(probe21.predicted_mean.to_numpy(), dtype=torch.float32)

flag_columns21 = [0, 3, 4, 5, 6, 7]
flag_masks21 = torch.zeros(64, 8)
for mask_id in range(64):
    for bit, column in enumerate(flag_columns21):
        flag_masks21[mask_id, column] = float((mask_id >> bit) & 1)
flag_vectors21 = flag_masks21 @ effect21

age_options21 = {
    "hours_log_div10": np.log1p(age21) / 10,
    "seconds_log_div10": np.log1p(age21 * 3600) / 10,
    "zero": np.zeros_like(age21),
    "hours_log": np.log1p(age21),
    "hours_linear_div10": age21 / 10,
}
grid21 = []
for age_name, age_values in age_options21.items():
    age_tensor = torch.tensor(age_values, dtype=torch.float32)
    pre = (prebase21[None, :, :]
           + z21[None, :, None] * effect21[2]
           + age_tensor[None, :, None] * effect21[1]
           + has21[None, :, None] * flag_vectors21[:, None, :])
    logits = F.gelu(pre) @ w221.T + b221 + transition21[None, :, :]
    probabilities = torch.softmax(logits, dim=-1)
    means = (probabilities * centres21[None, :, :]).sum(-1)
    errors = means - reference21[None, :]
    nlls = -torch.log(probabilities.gather(
        2, observed_bins21[None, :, None].expand(64, -1, 1)
    ).squeeze(-1).clamp_min(1e-30)).mean(1)
    for mask_id in range(64):
        grid21.append({
            "age": age_name,
            "flag_columns": tuple(c for bit, c in enumerate(flag_columns21)
                                  if (mask_id >> bit) & 1),
            "rms": float(errors[mask_id].square().mean().sqrt()),
            "bias": float(errors[mask_id].mean()),
            "nll": float(nlls[mask_id]),
            "mask_id": mask_id,
        })

grid21 = pd.DataFrame(grid21).sort_values("rms")
print("Captured current register example:", lf21[:5].numpy())
print("Reference NLL on probe:", probe21.nll.mean())
print("Best register candidates:")
print(grid21.head(15).to_string(index=False))
print("Current helper RMS:", np.sqrt(np.mean(
    (baseline21 - probe21.predicted_mean.to_numpy()) ** 2
)))

with open("/work/notes.md", "a") as f:
    f.write("\n## Register-flag search — Python 21; forward 4\n"
            "- Tested 64 binary flag layouts and five age encodings without changing weights.\n"
            "- Original rollout responses are weak relative to record-based impulse estimates.\n")
    f.write(grid21.head(8).to_string(index=False) + "\n")
```