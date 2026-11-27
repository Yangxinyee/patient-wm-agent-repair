The second cohort passes narrowly (+0.00473 nats); the combined penalty across 200 stays is about +0.00396. I’ll preserve this validated edit, then modestly strengthen insulin transmission and slightly reduce norepinephrine transmission using differences between saved weight snapshots. Better rollout agreement without exceeding the factual-fit limit would support keeping the change.

```python
from pathlib import Path
import torch, numpy as np, pandas as pd, hashlib, inspect

architecture299 = Path("/model/ARCHITECTURE.md").read_text()
notes_before299 = Path("/work/notes.md").read_text()

def tensor299(x):
    return torch.as_tensor(x).detach().cpu().to(torch.float32).clone()

names299 = list(edited())
backup299 = {n: tensor299(load(n)) for n in names299}
metadata299 = {
    "validated_candidate": (3.0, 0.0, 27.0, -0.5),
    "factual_first100_delta": 0.00319,
    "factual_second100_delta": 0.00473,
    "factual_pooled_delta": (4903 * 0.00319 + 4915 * 0.00473) / 9818,
    "rollout_first100": {
        "insulin_x2": -11.158, "insulin_half": 5.855,
        "ne_x2": 5.001, "ne_half": -4.793
    }
}
print("VALIDATED_BACKUP", metadata299)

b294 = globals().get("backup294")
b297 = globals().get("backup297")
print("SNAPSHOT_TYPES", type(b294).__name__, type(b297).__name__)
assert isinstance(b294, dict) and isinstance(b297, dict), "Preserved backup299; missing calibration snapshots."
assert all(n in b294 and n in b297 for n in names299), "Preserved backup299; incomplete snapshots."

# Notebook labels:
# backup294: insulin 3/0, gate 26, earlier-NE gain +1.
# backup297: insulin 3/0, gate 27, earlier-NE gain +0.5.
# backup299: insulin 3/0, gate 27, earlier-NE gain -0.5.
# Isolate the linear NE-gain difference, then subtract it to isolate
# the insulin-gate difference. No gradients or weight optimiser.
ne_delta299 = {n: tensor299(b297[n]) - backup299[n] for n in names299}
gate_raw299 = {
    n: tensor299(b294[n]) - backup299[n] - 1.5 * ne_delta299[n]
    for n in names299
}
gate_delta299 = {
    n: torch.where(gate_raw299[n].abs() > 3e-5,
                   gate_raw299[n], torch.zeros_like(gate_raw299[n]))
    for n in names299
}
gate_profile299 = {
    n: {"count": int(torch.count_nonzero(gate_delta299[n])),
        "max_abs": float(gate_delta299[n].abs().max())}
    for n in names299 if torch.count_nonzero(gate_delta299[n])
}
ne_profile299 = {
    n: {"count": int((ne_delta299[n].abs() > 3e-5).sum()),
        "max_abs": float(ne_delta299[n].abs().max())}
    for n in names299 if ne_delta299[n].abs().max() > 3e-5
}
print("NE_GAIN_DIFFERENCE", ne_profile299)
print("ISOLATED_GATE_DIFFERENCE", gate_profile299)

gate_ok299 = (
    set(gate_profile299) == {"z_post.0.bias"}
    and 0 < gate_profile299["z_post.0.bias"]["count"] <= 16
)
if gate_ok299:
    # Nominal gate 26.6 and NE gain -0.45; insulin gains unchanged.
    candidate299 = {
        n: backup299[n] + 0.4 * gate_delta299[n] + 0.05 * ne_delta299[n]
        for n in names299
    }
    candidate_label299 = (3.0, 0.0, 26.6, -0.45)
else:
    # Do not extrapolate an unaudited gate direction.
    candidate299 = {
        n: backup299[n] + 0.05 * ne_delta299[n] for n in names299
    }
    candidate_label299 = "NE gain -0.45 only; gate reconstruction rejected"

for n, t in candidate299.items():
    assert tuple(t.shape) == tuple(SHAPES[n]) and torch.isfinite(t).all()
for n, t in candidate299.items():
    save_edit(n, t)
print("SAVED_CANDIDATE", candidate_label299)
print("GATE_AUDIT_PASSED", gate_ok299)
print("EDITED_TENSORS", edited())

for folder, kind in [("fwd_45", "factual"), ("fwd_47", "factual"),
                     ("fwd_46", "dose_edit")]:
    p = Path("/work") / folder / (kind + "_edited.csv")
    frame = pd.read_csv(p)
    print("OUTPUT_SCHEMA", str(p), frame.columns.tolist())
    print(frame.head(1).to_string(index=False))

# Recover bounded identification context for the final notebook entry.
blocks299 = notes_before299.split("\n## ")
evidence299 = [
    b for b in blocks299
    if any(s in b.lower() for s in
           ["regression discontinu", "identification assumption",
            "distributed lag", "kernel_bootstrap", "instrumental"])
]
print("RECORD_EVIDENCE_CONTEXT")
print("\n\n".join(evidence299[:3] + evidence299[-2:])[:6500])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n\n## GPU passes 44–46; Run 299\n"
        "Validated insulin 3/0, gate 27, earlier-NE -0.5 (tool hash c198f94cd7). "
        "First-100 tuning factual penalty +0.00319; second-100 +0.00473; "
        f"reading-weighted pooled penalty {metadata299['factual_pooled_delta']:+.6f}. "
        "First-100 reached-row shifts: insulin -11.158/+5.855, NE +5.001/-4.793. "
        "Preserved all current edited tensors in backup299. "
        "Used labeled saved snapshots to isolate an NE-gain direction and "
        "audit the insulin-gate direction; no gradients or optimiser. "
        f"Gate audit passed={gate_ok299}; saved provisional candidate {candidate_label299}. "
        "Hypothesis: modest gate reduction strengthens insulin, while weaker reversed "
        "NE transmission reduces halving overshoot. GPU factual and rollout checks pending. "
        "Record-derived effect estimates remain estimates under identification assumptions; "
        "these calibration trials provide no new evidence of simulator truth.\n"
    )
```