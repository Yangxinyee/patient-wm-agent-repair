The head helper can read exported states directly. I’ll align the probe with its original-state cache and measure how much the continuous dose/rate perturbation changes predicted values. A clear, reproducible head-feature change would support a head-only repair; negligible or nonspecific changes would make that route doubtful.

```python
import json, numpy as np, pandas as pd, pathlib

with np.load("/work/fwd_8/states_edited.npz") as z:
    he_probe_states = {k: z[k].copy() for k in z.files}
print("Aligned stays:", np.array_equal(he_s0["stay_id"], he_probe_states["stay_id"]))
print("Aligned hours:", np.array_equal(he_s0["hour"], he_probe_states["hour"]))
assert np.array_equal(he_s0["stay_id"], he_probe_states["stay_id"])
assert np.allclose(he_s0["hour"], he_probe_states["hour"])

he_probe_head = ce_make_heads(he_records, he_probe_states)
assert he_probe_head["rows"][["stay_id", "hour", "event"]].equals(
    he_base["rows"][["stay_id", "hour", "event"]])

he_phi_probe = np.asarray(he_probe_head["phi"], dtype=np.float64)
he_dphi = he_phi_probe - he_phi0
he_probe_p = np.asarray(he_probe_head["probs"])
he_probe_mean = (he_probe_p * he_centres).sum(axis=1)
he_probe_nll = -np.log(np.maximum(he_probe_p[np.arange(len(he_bins)), he_bins], 1e-30))

he_probe_diag = he_base["rows"][["stay_id", "hour", "event"]].copy()
he_probe_diag["mean_shift"] = he_probe_mean - he_mean0
he_probe_diag["nll_shift"] = he_probe_nll - he_nll0
he_probe_diag["feature_shift_norm"] = np.linalg.norm(he_dphi, axis=1)
print("\nPROBE HEAD CHANGES, ALL HOURS:")
print(he_probe_diag.groupby("event").agg(
    n=("mean_shift", "size"), mean_shift=("mean_shift", "mean"),
    rms_shift=("mean_shift", lambda v: np.sqrt(np.mean(v*v))),
    nll_shift=("nll_shift", "mean"),
    feature_norm=("feature_shift_norm", "mean")).to_string())
print("\nPROBE HEAD CHANGES, HOURS 13–24:")
print(he_probe_diag.query("13 <= hour <= 24").groupby("event")[
    ["mean_shift", "nll_shift", "feature_shift_norm"]].mean().to_string())

print("\nRECORD-BASED RESPONSE KERNELS:")
for key, value in he_kernels.items():
    print(key, np.round(value, 5))
print("\nPANELS:")
for name in ["P", "R", "L", "Age"]:
    obj = globals().get(name, {})
    print(name, [(k, getattr(v, "shape", None)) for k, v in list(obj.items())[:8]])
print("\nDRUG RECORD SAMPLE:")
print(train[train.event.isin(["insulin", "norepinephrine"])].head(8).to_string(index=False))

print("\nDATA DOCUMENTATION:")
print(pathlib.Path("/data/README.md").read_text()[:2600])
print("\nFORWARD SUMMARIES:")
for k in [1, 3, 5, 6]:
    print(k, pathlib.Path(f"/work/fwd_{k}/summary.json").read_text()[:900])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 144\nAligned original and numeric-probe exported states and reconstructed their value heads. Comparing sensitivity and factual drift before choosing a constrained edit. No edits remain. Counts: 144 Python runs, 8 forwards.\n")
```