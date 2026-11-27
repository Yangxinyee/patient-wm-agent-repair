The assembled rank-two correction is available, and the replay confirms that it maps directly to `core.b_in.weight`. I’ll test it at half strength, removing the earlier latent edit that failed rollout validation. This is a conservative first test: factual NLL must remain within 0.005 nats, and the rollout must strengthen the record-supported drug effects.

```python
import numpy as np, pandas as pd, torch
from pathlib import Path

print("=== CANDIDATE CALIBRATION ===")
for r251_name in ["r245_need", "r245_response", "r245_Q", "r245_R"]:
    print(r251_name, globals().get(r251_name))
print("Original local dose means:", r245_original_means)

r251_previous = {
    name: np.asarray(load(name), dtype=np.float32).copy()
    for name in edited()
}
np.savez("/work/pre_recurrent_candidate.npz", **r251_previous)

r251_C = np.asarray(r245_C, dtype=np.float32)
r251_base = np.asarray(load_original("core.b_in.weight"), dtype=np.float32)
assert r251_C.shape == r251_base.shape
assert np.isfinite(r251_C).all()
r251_strength = 0.5
r251_weight = r251_base + r251_strength * r251_C

for r251_name in list(edited()):
    drop_edit(r251_name)
save_edit("core.b_in.weight", r251_weight)

r251_tuning = pd.read_csv("/data/tuning.csv")
r251_stays = sorted(r251_tuning.stay_id.unique().tolist())[:100]
r251_stays = [int(x) for x in r251_stays]
print("\n=== SUBMITTED CANDIDATE ===")
print("Edited tensors:", edited())
print("Correction strength:", r251_strength)
print("Original/correction norms:",
      float(np.linalg.norm(r251_base)), float(np.linalg.norm(r251_strength * r251_C)))
print("Tuning stays:", r251_stays)

with open("/work/notes.md", "a") as r251_file:
    r251_file.write(
        "\n## Run 251: submit half-strength recurrent candidate\n"
        "- Replaced the causally ineffective six-tensor latent candidate with rank-two core.b_in.weight correction.\n"
        "- Strength 0.5; original latent weights restored. Previous candidate archived in /work/pre_recurrent_candidate.npz.\n"
        "- Local factual shifts are modest, but official factual NLL and multi-hour rollout are not yet validated.\n"
        "- Python runs: 251; forward passes: 17.\n"
    )
```
