The rollout still has the wrong balance: insulin doubling shifts glucose by −10.39 mg/dL, but halving shifts it by +12.63; norepinephrine doubling remains weak at +0.73 mmHg. I’ll check the record-based targets and dose-channel construction to distinguish transmission asymmetry from recurrent-feedback distortion before increasing any response strength.

```python
from pathlib import Path
import re, inspect, numpy as np, pandas as pd, torch

rollout212 = pd.read_csv("/work/fwd_21/dose_edit_edited.csv")
eligible212 = rollout212[rollout212["reached"].astype(bool)]
print("=== CURRENT ROLLOUT AGGREGATES ===")
for label212, frame212 in eligible212.groupby("edit"):
    print(label212,
          "row_shift", round(frame212["shift"].mean(), 4),
          "equal_stay_shift", round(frame212.groupby("stay_id")["shift"].mean().mean(), 4),
          "generated_baseline", round(frame212["factual"].mean(), 4))

print("=== EARLIER RECORD-IDENTIFICATION NOTES ===")
text212 = Path("/work/notes.md").read_text()
for number212 in [6, 7, 8, 9, 10, 11, 12, 14, 24, 25]:
    match212 = re.search(rf"(?ms)^## Run {number212}\b.*?(?=^## |\Z)", text212)
    if match212:
        print(match212.group(0)[:390])

print("=== RETAINED CHANNEL CONSTRUCTORS ===")
functions212 = [
    (k, v) for k, v in list(globals().items())
    if callable(v) and re.search(r"(dual|gated|channel|branch|joint|calib)", k, re.I)
]
for name212, fn212 in sorted(functions212)[:24]:
    try:
        print(name212, inspect.signature(fn212), "globals", fn212.__code__.co_names[:20])
    except (ValueError, AttributeError, TypeError):
        pass

print("=== POSTERIOR EDIT STRUCTURE ===")
for name212 in ["z_post.0.weight", "z_post.2.weight", "z_in.weight"]:
    now212 = np.asarray(load(name212))
    orig212 = np.asarray(load_original(name212))
    axis212 = 1 if name212 == "z_post.0.weight" else 0
    norms212 = np.linalg.norm(now212 - orig212, axis=axis212)
    changed212 = np.flatnonzero(norms212 > 1e-5)
    print(name212, "shape", now212.shape, "changed indices", changed212.tolist())
    print("largest changes",
          [(int(i), round(float(norms212[i]), 5))
           for i in np.argsort(norms212)[-8:][::-1] if norms212[i] > 1e-5])
print("raw_candidates87 keys", list(raw_candidates87) if isinstance(raw_candidates87, dict) else type(raw_candidates87))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 212; forward passes used 20/50\n"
        "Ridge-10 NE connection scale -1 passed GPU factual on 98 tuning stays: "
        "original 1.59352, edited 1.59702, delta .00350. Paired dose rollout row shifts: "
        "insulin x2 -10.385, half +12.633; NE x2 +.730, half -3.130. "
        "Both channels retain pronounced half/double asymmetry; inspecting causal evidence "
        "and dose transmission before changing strengths. No weights changed in this run.\n"
    )
```