Each original export covers half of the probe stays; together they appear to cover every bundle. I’ll combine them and verify the match before comparing head features. That avoids spending a forward pass on an export we may already have.

```python
import numpy as np, pathlib

hp_parts = []
for k in [2, 4]:
    with np.load(f"/work/fwd_{k}/states_original.npz") as z:
        hp_parts.append({name: z[name].copy() for name in z.files})
hp_union = {name: np.concatenate([s[name] for s in hp_parts], axis=0)
            for name in hp_parts[0]}
hp_lookup = {key: i for i, key in enumerate(hp_keys(hp_union))}
assert len(hp_lookup) == len(hp_union["stay_id"])
hp_take = np.array([hp_lookup[key] for key in hp_probe_keys])
hp_original_states = {name: value[hp_take] for name, value in hp_union.items()}
print("Complete match:", len(hp_take),
      "maximum hour error:", np.max(np.abs(hp_original_states["hour"] -
                                           he_probe_states["hour"])))

hp_records = train[train.stay_id.isin(hp_probe_ids)].copy()
hp_original_head = ce_make_heads(hp_records, hp_original_states)
hp_probe_head = ce_make_heads(hp_records, he_probe_states)
assert hp_original_head["rows"][["stay_id", "hour", "event"]].equals(
    hp_probe_head["rows"][["stay_id", "hour", "event"]])

hp_phi0 = np.asarray(hp_original_head["phi"], dtype=np.float64)
hp_dphi = np.asarray(hp_probe_head["phi"], dtype=np.float64) - hp_phi0
hp_p0 = np.asarray(hp_original_head["probs"])
hp_pp = np.asarray(hp_probe_head["probs"])
hp_centres = np.asarray(hp_original_head["centres"])
hp_bins = np.asarray(hp_original_head["truebin"], dtype=int)
hp_mean0 = (hp_p0 * hp_centres).sum(1)
hp_meanp = (hp_pp * hp_centres).sum(1)
hp_nll0 = -np.log(np.maximum(hp_p0[np.arange(len(hp_bins)), hp_bins], 1e-30))
hp_nllp = -np.log(np.maximum(hp_pp[np.arange(len(hp_bins)), hp_bins], 1e-30))
hp_diag = hp_original_head["rows"][["stay_id", "hour", "event"]].copy()
hp_diag["mean_shift"] = hp_meanp - hp_mean0
hp_diag["nll_shift"] = hp_nllp - hp_nll0
hp_diag["feature_norm"] = np.linalg.norm(hp_dphi, axis=1)

print("\nALL-HOUR PROBE CHANGES:")
print(hp_diag.groupby("event").agg(
    n=("mean_shift", "size"), shift=("mean_shift", "mean"),
    rms=("mean_shift", lambda v: np.sqrt(np.mean(v*v))),
    nll=("nll_shift", "mean"), feature_norm=("feature_norm", "mean")).to_string())
print("\nHOURS 13–24:")
print(hp_diag.query("13 <= hour <= 24").groupby("event")[
    ["mean_shift", "nll_shift", "feature_norm"]].mean().to_string())

print("\nHEAD WEIGHT CACHE CHECK:")
for name in NAMES:
    if name.startswith("attr_heads.value_bin") and name in W:
        print(name, float(np.max(np.abs(np.asarray(W[name]) -
                                       np.asarray(load_original(name))))))
print("\nDRUG NORMALIZATION:")
for drug in ["insulin", "norepinephrine"]:
    for attr in ["dose", "rate"]:
        meta = ce_meta[(drug, attr)]
        print(drug, attr, {k: meta[k] for k in ["median", "iqr"]})
print("\nDRUG RECORD SAMPLE:")
print(train[train.event.isin(["insulin", "norepinephrine"])].head(6).to_string(index=False))
print("\nDATA DOCUMENTATION:")
print(pathlib.Path("/data/README.md").read_text()[:2200])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 146\nCombined original exports 2 and 4 to align all 100 numeric-probe stays. Reconstructed paired value-head features; no recurrent replay required and no weights edited. Counts: 146 Python runs, 8 forwards.\n")
```