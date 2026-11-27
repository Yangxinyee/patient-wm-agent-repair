The last rollout preserved the insulin improvement: doubling shifted glucose by −11.017 versus −1.879 originally. Norepinephrine doubling now has the record-supported sign, but its response remains too small. I’ll keep C rather than make an unvalidated change. This final audit will confirm the submitted weights and factual evidence; any mismatch would invalidate that choice.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import hashlib
import json

def final_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value, dtype=np.float32)

final_weight = final_numpy(load("core.b_in.weight"))
original_weight = final_numpy(load_original("core.b_in.weight"))
with np.load("/work/joint291.npz", allow_pickle=False) as archive:
    final_delta = archive["delta"].copy()
    final_metadata = {
        key: archive[key].item()
        for key in ["insulin_scale", "insulin_blend", "ne_scale", "source_scale"]
    }

reconstructed = np.asarray(original_weight + final_delta, dtype=np.float32)
assert np.allclose(final_weight, reconstructed, rtol=2e-6, atol=2e-6)
assert set(edited()) == {"core.b_in.weight"}
final_sha = hashlib.sha256(final_weight.tobytes()).hexdigest()
assert final_sha == (
    "c93a8cf4fad68f4cec97c3f7c5cc1c2ceea479611ba573bd115eb6df4f99bbc2"
)

paired_readings = []
for request in [48, 49]:
    original_frame = pd.read_csv(
        f"/work/fwd_{request}/factual_original.csv"
    )
    edited_frame = pd.read_csv(
        f"/work/fwd_{request}/factual_edited.csv"
    )
    identity = ["stay_id", "hour", "event"]
    assert original_frame[identity].equals(edited_frame[identity])
    paired = original_frame[identity].copy()
    paired["original_nll"] = original_frame["nll"].to_numpy()
    paired["edited_nll"] = edited_frame["nll"].to_numpy()
    paired["delta_nll"] = (
        paired["edited_nll"] - paired["original_nll"]
    )
    paired_readings.append(paired)

final_factual = pd.concat(paired_readings, ignore_index=True)
factual_report = {
    "n_stays": int(final_factual["stay_id"].nunique()),
    "n_readings": len(final_factual),
    "original_nll": float(final_factual["original_nll"].mean()),
    "edited_nll": float(final_factual["edited_nll"].mean()),
    "increase": float(final_factual["delta_nll"].mean()),
}
assert factual_report["increase"] < 0.005

# These values are the supplied forward50 summary, not estimates of hidden truth.
final_rollout_report = {
    "insulin_x2_glucose_finger": -11.017,
    "insulin_half_glucose_finger": 5.130,
    "ne_x2_map_arterial": 0.363,
    "ne_half_map_arterial": -0.741,
}
for model in ["original", "edited"]:
    path = Path(f"/work/fwd_50/dose_edit_{model}.csv")
    assert path.is_file()
    frame = pd.read_csv(path)
    print("Final rollout artifact:", model, len(frame), list(frame.columns))

print("FINAL CHOICE: keep candidate C unchanged")
print("Edited tensors:", edited())
print("Candidate metadata:", final_metadata)
print("SHA256:", final_sha)
print("Maximum reconstruction difference:",
      float(np.max(np.abs(final_weight - reconstructed))))
print("Measured factual validation:", factual_report)
print("Forward50 measured shifts:", final_rollout_report)

with Path("/work/notes.md").open("a") as handle:
    handle.write(
        "\n## Run 296: final choice and submission audit\n"
        "Forward50 exact paired rollout on tuning100..199: original insulin x2 "
        "-1.879, half +0.868; NE x2 -0.312, half -0.647. "
        "Candidate C: insulin x2 -11.017, half +5.130; "
        "NE x2 +0.363, half -0.741. "
        "Keep C unchanged: main insulin correction retained; "
        "NE doubling sign repaired, but NE magnitudes remain substantially "
        "below record-derived estimates. No further unvalidated edits.\n"
        "Final factual evidence: "
        + json.dumps(factual_report)
        + ". This is measured on selected200 tuning stays, not unseen "
        "evaluation stays; all400 C factual fit was not measured. "
        "B all400 and C's shrinkage support, but do not guarantee, safety.\n"
        "Only core.b_in.weight edited, original plus archived joint291 delta; "
        "closed-form record-derived correction, no gradients or optimizer. "
        "Final SHA256: " + final_sha + ". "
        "Python runs: 296; forwards: 50. Ready for final submission.\n"
    )
```