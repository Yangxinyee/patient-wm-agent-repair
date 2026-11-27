The installed weights exactly match the validated checkpoint. I’ll test gate strength 150 instead of 180: its cached factual cost was slightly lower, with unchanged local contrasts. Preserved doubling and improved halving on the GPU would support the trial; otherwise I’ll restore the fallback.

```python
import pandas as pd
from pathlib import Path

smooth150_checkpoint = {
    n: np.asarray(v, dtype=np.float32).copy()
    for n, v in gate_candidates274[(150.0, 1.0)].items()
}
assert set(smooth150_checkpoint) == set(validated_gated8_checkpoint)
for n, v in smooth150_checkpoint.items():
    assert v.shape == tuple(SHAPES[n])
    assert np.isfinite(v).all()
    save_edit(n, v)

tune_ids_final = pd.read_csv(
    "/data/tuning.csv", usecols=["stay_id"]
)["stay_id"].drop_duplicates().tolist()
print("First100 tuning IDs:", tune_ids_final[:100])
print("Installed smoother-gate trial; protected fallback unchanged.")
print("Edited tensors:", edited())

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Smoother-gate trial; Python 277, forwards 33/50\n"
        "- Verified exact equality with validated_gated8_checkpoint.\n"
        "- Installed cached strength150/amplitude1 candidate, versus validated strength180.\n"
        "- CPU tuning NLL difference -.000148; local contrasts unchanged.\n"
        "- UNACCEPTED pending exact GPU rollout and factual checks; fallback protected.\n"
    )
```
