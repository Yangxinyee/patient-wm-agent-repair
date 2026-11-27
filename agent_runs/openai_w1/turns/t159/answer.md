The state rows are not in the same order, so the attempted comparison stopped before producing any sensitivity result. I’ll match exports by stay and bundle time rather than row number. Complete one-to-one matches would resolve this as an ordering issue; missing bundles would require a new paired export.

```python
import numpy as np, pathlib, json

def hp_keys(st):
    return [(int(s), int(round(float(h) * 3600)))
            for s, h in zip(st["stay_id"], st["hour"])]

hp_probe_keys = hp_keys(he_probe_states)
hp_probe_ids = np.unique(he_probe_states["stay_id"])
print("Probe:", len(hp_probe_keys), len(hp_probe_ids), hp_probe_ids[:8])
hp_source = None
for k in [2, 4, 7]:
    with np.load(f"/work/fwd_{k}/states_original.npz") as z:
        st = {name: z[name].copy() for name in z.files}
    keys = hp_keys(st)
    lookup = {key: i for i, key in enumerate(keys)}
    matched = sum(key in lookup for key in hp_probe_keys)
    overlap = len(set(st["stay_id"]) & set(hp_probe_ids))
    print("Original export", k, "rows", len(keys), "stay overlap", overlap,
          "matched bundles", matched, "duplicate keys", len(keys)-len(lookup))
    if matched == len(hp_probe_keys) and len(lookup) == len(keys):
        take = np.array([lookup[key] for key in hp_probe_keys])
        hp_original_states = {name: value[take] for name, value in st.items()}
        hp_source = k

if hp_source is not None:
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
    print("\nMatched original export:", hp_source)
    print(hp_diag.groupby("event").agg(
        n=("mean_shift", "size"), shift=("mean_shift", "mean"),
        rms=("mean_shift", lambda v: np.sqrt(np.mean(v*v))),
        nll=("nll_shift", "mean"), feature_norm=("feature_norm", "mean")).to_string())
else:
    print("No complete match; no head comparison performed.")

print("\nResponse kernels:")
for key, value in he_kernels.items():
    print(key, np.round(value, 5))
print("\nPanel names:")
for name in ["P", "R", "L", "Age"]:
    obj = globals().get(name, {})
    print(name, [(k, getattr(v, "shape", None)) for k, v in list(obj.items())[:8]])

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 144–145\nRun 144 stopped on state-row alignment before any comparison. Run 145 matches by stay and bundle time and reconstructs heads only on complete matches. No edits. Counts: 145 Python runs, 8 forwards.\n")
```