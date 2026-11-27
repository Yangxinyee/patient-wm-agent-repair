The retained innovation decoders had poor dose extrapolation, especially for norepinephrine, so I won’t reuse them. I’ll rebuild bundles directly from the records with the verified units. Matching exported atom counts and posterior states will confirm that the new calibration uses the model’s actual inputs.

```python
import torch.nn.functional as F
from collections import defaultdict

rr_numeric = {
    (entry["code"], entry["attribute"]): entry
    for entry in rr_vocab["numeric"]
}
rr_codes = {entry["name"]: entry["id"] for entry in rr_vocab["codes"]}
rr_drugs = {"insulin", "norepinephrine"}
rr_sensors = {"glucose_finger", "glucose_lab", "map_arterial", "map_cuff"}
rr_units = {"glucose_finger": 0, "glucose_lab": 5,
            "map_arterial": 2, "map_cuff": 2}

def rr_linear(x, prefix, weights=None):
    w = rr_original if weights is None else weights
    return F.linear(x, w[prefix + ".weight"], w.get(prefix + ".bias"))

def rr_mlp(x, prefix, activation="gelu", weights=None):
    y = rr_linear(x, prefix + ".0", weights)
    y = F.gelu(y) if activation == "gelu" else F.silu(y)
    return rr_linear(y, prefix + ".2", weights)

def rr_norm(x, prefix, weights=None):
    w = rr_original if weights is None else weights
    return F.layer_norm(x, (192,), w[prefix + ".weight"],
                        w[prefix + ".bias"], 1e-5)

def rr_atom(code, value=np.nan, rate=np.nan, dose=np.nan, end=False):
    nums = np.zeros(6, dtype=np.float32)
    bins = {"value": 0, "dose": 0, "rate": 0}
    for attr, val, col in [("value", value, 0), ("dose", dose, 2),
                           ("rate", rate, 4)]:
        if not end and np.isfinite(val):
            spec = rr_numeric[(code, attr)]
            nums[col] = np.clip((val - spec["median"]) / spec["iqr"], -10, 10)
            nums[col + 1] = 1
            bins[attr] = 1 + np.searchsorted(
                spec["fine_bin_edges"], val, side="left"
            )
    ekind = (2 if code in rr_drugs else 1 if code in rr_sensors
             else 3 if code == "hospital_admission" else 4)
    v = (
        rr_original["embed.code.weight"][rr_codes[code]]
        + rr_original["embed.kind.weight"][2 if end else 0]
        + rr_original["embed.ekind.weight"][ekind]
        + rr_linear(torch.from_numpy(nums), "embed.numeric")
    )
    for attr in bins:
        v = v + rr_original[f"embed.bins.{attr}.weight"][bins[attr]]
    if not end and code in rr_sensors:
        v = v + rr_original["embed.categorical.unit.weight"][rr_units[code]]
    if not end and code in rr_drugs:
        v = v + rr_original["embed.categorical.rate_unit.weight"][
            2 if code == "insulin" else 3
        ]
        v = v + rr_original["embed.categorical.unit.weight"][
            3 if code == "insulin" else 4
        ]
    for table in ["embed.kq.weight", "embed.text.weight"]:
        if table in rr_original:
            v = v + rr_original[table][0]
    return v

rr_stays4 = set(map(int, np.unique(rr_state4["stay_id"])))
rr_records4 = rr_train[rr_train.stay_id.isin(rr_stays4)].copy()
rr_bundle_map4 = defaultdict(list)
for row in rr_records4.itertuples(index=False):
    key = (int(row.stay_id), int(round(float(row.hour) * 60)))
    rr_bundle_map4[key].append({
        "code": row.event, "value": row.value, "rate": row.rate,
        "dose": row.dose, "end": False
    })
    if np.isfinite(row.end_hour):
        endkey = (int(row.stay_id), int(round(float(row.end_hour) * 60)))
        rr_bundle_map4[endkey].append({
            "code": row.event, "value": np.nan, "rate": np.nan,
            "dose": np.nan, "end": True
        })

rr_keys4 = [
    (int(s), int(round(float(h) * 60)))
    for s, h in zip(rr_state4["stay_id"], rr_state4["hour"])
]
rr_bundles4 = [rr_bundle_map4[key] for key in rr_keys4]
rr_counts4 = np.array([len(bundle) for bundle in rr_bundles4])
rr_bad4 = np.flatnonzero(rr_counts4 != rr_state4["n_atoms"])
print("=== BUNDLE ALIGNMENT ===")
print("rows", len(rr_keys4), "count disagreements", len(rr_bad4))
for i in rr_bad4[:8]:
    print(rr_keys4[i], "constructed", rr_counts4[i],
          "exported", rr_state4["n_atoms"][i],
          [(a["code"], a["end"]) for a in rr_bundles4[i]])

rr_owner4, rr_desc4, rr_atom_rows4 = [], [], []
for i, bundle in enumerate(rr_bundles4):
    for a in bundle:
        rr_owner4.append(i)
        rr_desc4.append(a)
        rr_atom_rows4.append(rr_atom(**a))
rr_owner4 = torch.tensor(rr_owner4, dtype=torch.long)
rr_atoms4 = torch.stack(rr_atom_rows4)
rr_is_drug4 = np.array([a["code"] in rr_drugs for a in rr_desc4])
rr_is_end4 = np.array([a["end"] for a in rr_desc4])
print("atoms", tuple(rr_atoms4.shape))
print("=== GAP TENSORS ===")
for name in NAMES:
    if "gap" in name:
        print(name, tuple(rr_original[name].shape))

print("=== ORIGINAL REPRESENTATIVE ROLLOUT ===")
print(Path("/work/fwd_5/summary.json").read_text()[:2500])

with Path("/work/notes.md").open("a") as f:
    f.write(
        f"\n## Run 102\nFresh atom/bundle reconstruction with verified unit IDs. "
        f"Export4 atom-count disagreements: {len(rr_bad4)} / {len(rr_keys4)}. "
        "Rejected obsolete innovation decoders because their dose extrapolation was "
        "distorted. No model weights edited.\n"
        "Counts: 102 Python runs, 6 forward passes.\n"
    )
```
