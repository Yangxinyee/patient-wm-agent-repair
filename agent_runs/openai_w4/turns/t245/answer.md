Both drugs show the same imbalance: doubling is too weak relative to halving. That suggests checking how off-policy doses are encoded, rather than simply multiplying both response strengths. I’ll recover the cached dose-augmentation tests and causal estimates; those tests can separate encoding errors from errors introduced later by feedback.

```python
from pathlib import Path
import re, inspect, numpy as np, torch

print("=== SMALL-MODEL / REPLAY FUNCTIONS ===")
for name213, fn213 in sorted(list(globals().items())):
    if not callable(fn213) or not hasattr(fn213, "__code__"):
        continue
    if (re.search(r"(87|90|94|102|173|174|180|185|186|192|193)$", name213)
            or re.search(r"(augment|innovation|raw|dose|encode|embed|policy|panel)", name213)):
        try:
            print(name213, inspect.signature(fn213),
                  "globals:", fn213.__code__.co_names[:12])
        except (ValueError, TypeError):
            pass

print("=== CACHED RAW-INPUT ARRAYS ===")
for name213, value213 in sorted(list(globals().items())):
    if isinstance(value213, (np.ndarray, torch.Tensor)):
        if value213.ndim == 2 and value213.shape[1] in [576, 577]:
            print(name213, tuple(value213.shape), str(value213.dtype))
    elif isinstance(value213, dict) and re.search(r"(87|174|180|185|186|192|193)$", name213):
        print(name213, "keys:", list(value213)[:15])

print("=== RECORD-EFFECT AND EARLIER INSULIN RESULTS ===")
text213 = Path("/work/notes.md").read_text()
sections213 = re.split(r"(?m)(?=^## )", text213)
selected_sections213 = []
for section213 in sections213:
    if re.search(r"(dynamic|autoregress|instrument|bootstrap|causal estimates|effect estimates|rollout target|RD.*IV)", section213, re.I):
        selected_sections213.append(section213)
for section213 in selected_sections213[:8]:
    print(section213[:520])
for number213 in [160, 166, 174, 179, 180]:
    match213 = re.search(rf"(?ms)^## Run {number213}\b.*?(?=^## |\Z)", text213)
    if match213:
        print(match213.group(0)[:470])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 213\n"
        "Current ridge-10 rollout equal-stay shifts are insulin x2 -8.722, half +11.677; "
        "NE x2 +.609, half -2.840. Row-weighted shifts differ materially, so preserve both. "
        "Recovering cached dose-augmentation tests and record-effect estimates to distinguish "
        "input transmission from recurrent distortion. No weights changed.\n"
    )
```