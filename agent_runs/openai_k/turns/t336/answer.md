The exact pooled increase is +0.004253, but the stay-bootstrap upper bound reaches +0.005865. That is too close to the eligibility limit for comfort. I’ll reduce both corrections, keeping MAP stronger than in the validated fallback. Improved factual margin with an insulin response near the record estimate would support this compromise.

```python
import numpy as np
import torch
from pathlib import Path

with np.load("/work/validated_current400.npz") as a288:
    original288 = a288["original"].copy()
    glucose288 = a288["glucose_correction"].copy()
    map288 = a288["map_correction"].copy()
    fallback288 = a288["core_b_in"].copy()

assert np.allclose(
    fallback288, original288 + glucose288 + map288, atol=3e-5, rtol=3e-5
)
with np.load("/work/balanced_candidate.npz") as a288:
    assert np.allclose(
        a288["core_b_in"],
        original288 + (1.08 / 1.15) * glucose288 + 1.6 * map288,
        atol=3e-5, rtol=3e-5,
    )

glucose_strength288 = 1.00
map_relative_strength288 = 1.40
candidate288 = (
    original288.astype(np.float64)
    + (glucose_strength288 / 1.15) * glucose288.astype(np.float64)
    + map_relative_strength288 * map288.astype(np.float64)
).astype(np.float32)

assert np.isfinite(candidate288).all()
assert candidate288.shape == tuple(SHAPES["core.b_in.weight"])
np.savez(
    "/work/margin_candidate.npz",
    core_b_in=candidate288,
    original=original288,
    glucose_correction=glucose288,
    map_correction=map288,
    glucose_strength=glucose_strength288,
    map_relative_strength=map_relative_strength288,
)
save_edit("core.b_in.weight", torch.from_numpy(candidate288.copy()))

print({
    "edited": edited(),
    "glucose_strength": glucose_strength288,
    "map_relative_to_validated_fallback": map_relative_strength288,
    "map_full_route_strength": 0.5 * map_relative_strength288,
    "weight_norm": float(np.linalg.norm(candidate288.astype(np.float64))),
    "status": "Unvalidated trial; both validated fallbacks remain archived.",
})

notes288 = Path("/work/notes.md").read_text().splitlines()
evidence288 = [
    line for line in notes288
    if any(term in line.lower() for term in
           ["convolution", "dose-regime", "linearity", "placebo", "negative control"])
]
print("\nRETAINED IDENTIFICATION CHECKS")
for line in evidence288[-12:]:
    print(line[:300])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 288: safety-margin candidate\n"
        "- Balanced candidate passes the observed 400-stay mean but its "
        "97.5% stay-bootstrap bound exceeds +0.005; not selected as final.\n"
        "- Submitted glucose strength 1.00 and MAP full-route strength 0.70 "
        "(1.40 times the validated fallback MAP component).\n"
        "- Goal: insulin x2 nearer record-derived target, stronger MAP response "
        "than fallback, and a safer factual-loss margin.\n"
        "- Archived in margin_candidate.npz; needs official factual and "
        "dose-rollout validation. No gradients or weight optimizer.\n"
        "- Python runs: 288; forward passes: 36.\n"
    )
```