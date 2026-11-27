The paired files confirm the 0.00614 increase. The helper allows insulin and MAP strength to change independently; the diagnostic error only affected table printing. I’ll retain insulin strength 0.90 and reduce MAP gain from 9.5 to 7, then retest the failing batch. Passing with margin would justify that tradeoff.

```python
import numpy as np
import torch

safe247_insulin_strength = 0.90
safe247_map_gain = 7.0

print("=== CACHED FACTUAL SCREEN ===")
safe247_rows = frontier242[
    np.isclose(frontier242["ins_strength"], safe247_insulin_strength)
    & np.isclose(frontier242["map_gain"], safe247_map_gain)
]
print(safe247_rows.to_string(index=False))

safe247_weights = make242(safe247_insulin_strength, safe247_map_gain)
assert set(safe247_weights) == set(head177_names)
for safe247_name, safe247_tensor in safe247_weights.items():
    assert tuple(safe247_tensor.shape) == tuple(load_original(safe247_name).shape)
    assert torch.isfinite(torch.as_tensor(safe247_tensor)).all()

np.savez(
    "/work/safer_candidate247.npz",
    **{k: torch.as_tensor(v).detach().cpu().numpy()
       for k, v in safe247_weights.items()}
)
for safe247_name, safe247_tensor in safe247_weights.items():
    save_edit(safe247_name, safe247_tensor)

print("Installed strengths:", safe247_insulin_strength, safe247_map_gain)
print("Edited tensors:", edited())

with open("/work/notes.md", "a") as safe247_f:
    safe247_f.write(
        "\n## Runs 246–247: independent validation failure and backoff\n"
        "- Forward 25 / fwd_26: candidate243 factual dNLL +0.00456 on tuning 0–99.\n"
        "- Forward 26 / fwd_27: rollouts insulin -10.198/+5.585, NE +1.059/-1.872.\n"
        "- Forward 27 / fwd_28: tuning 200–299 factual dNLL +0.006138; candidate243 not reliably validated.\n"
        "- Run 246 diagnostic sorting error did not change weights or write its planned note.\n"
        "- Run 247 installed insulin strength 0.90, MAP gain 7.0; saved safer_candidate247.npz.\n"
        "- Pending paired GPU factual check on the previously failing independent batch.\n"
        "- Forward passes used: 27; Python runs used: 247.\n"
    )
```