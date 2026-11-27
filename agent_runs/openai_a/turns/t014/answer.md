We now have states for the same stays used in the factual baseline. I’ll inspect the head and dose encodings so any edit uses the model’s actual representation. If dose information is not recoverable from that representation, a simple readout correction would be unreliable and we’ll need another approach.

```python
states_tune = np.load("/work/fwd_3/states_original.npz")
print("=== STATE ARRAYS ===")
for k in states_tune.files:
    arr = states_tune[k]
    print(k, arr.shape, arr.dtype)
print("first bundle metadata:",
      list(zip(states_tune["stay_id"][:12], states_tune["hour"][:12],
               states_tune["has_insulin"][:12],
               states_tune["has_norepinephrine"][:12],
               states_tune["n_atoms"][:12])))

print("\n=== HEAD/ENCODING TENSOR NAMES ===")
for name in NAMES:
    if not any(s in name for s in ["core.prelude", "core.blocks", "core.coda"]):
        print(name, SHAPES[name])

print("\n=== NUMERIC VOCAB DETAILS ===")
print("numeric_note:", VOCAB["numeric_note"])
print("numeric keys:", list(VOCAB["numeric"].keys()))
for k,v in VOCAB["numeric"].items():
    if isinstance(v, dict):
        print(k, "keys:", list(v.keys())[:18])
        for code in ["6","7","8","9"]:
            if code in v:
                print(k,code,v[code])
    else:
        print(k,str(v)[:1200])
print("value centres:", VOCAB["value_bin_centres"])

print("\n=== REGISTER WEIGHT COLUMN NORMS ===")
for name in ["last_numeric.weight","embed.numeric.weight"]:
    W=load_original(name)
    W=W.numpy() if hasattr(W,"numpy") else np.asarray(W)
    print(name,np.round(np.linalg.norm(W,axis=0),4).tolist())

print("\n=== OBSERVED DRUG RATES ===")
for drug in ["insulin","norepinephrine"]:
    rr=train.loc[train.event.eq(drug),"rate"]
    print(drug, "quantiles:",rr.quantile([0,.1,.5,.9,.99,1]).to_dict())
    print("most frequent:",rr.value_counts().head(12).to_dict())

with open("/work/notes.md","a") as f:
    f.write("\n## State export; Python 11, forward 3\n"
            "- Exported original states for first 100 tuning stays; no weights edited.\n"
            "- Preparing a head-feature reconstruction and dose-residual decoder before attempting a low-rank edit.\n"
            "- Correction: the forward readout is a pooled mean over reached stay-hours, not an equally weighted mean over stays.\n")
```