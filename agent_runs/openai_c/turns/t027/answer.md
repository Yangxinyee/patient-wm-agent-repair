Amplification helped insulin modestly, but the effects remain far below the record-based estimates; norepinephrine is still nearly unresponsive. I’ll inspect the retained state directions and earlier latent-pathway test before choosing a more targeted edit. A useful direction must carry dose variation without strongly disturbing ordinary factual states.

```python
import json, os, numpy as np, pandas as pd

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 16 / forwards 9–10: candidate C results\n"
            "- Numeric dose/rate gain 4: factual NLL 1.63584 versus 1.63461 (+0.00123).\n"
            "- Probe rollouts: insulin x2 -2.503, half +1.691; NE x2 +0.127, half -0.003.\n"
            "- Modest insulin improvement, no adequate NE response. Retained while investigating targeted directions.\n"
            "- Python runs: 16; forward passes: 10.\n")

_notes = open("/work/notes.md").read()
_start = _notes.find("## Run 10")
_end = _notes.find("## Runs 13")
print("=== LATENT-PATHWAY TEST NOTES ===")
print(_notes[_start:_end] if _start >= 0 and _end > _start else "No matching section")

print("=== VARIABLES (names only) ===")
print(", ".join(k for k in list(globals()) if not k.startswith("_")))

def compact_structure(x, depth=0):
    if isinstance(x, dict):
        return {str(k): compact_structure(v, depth+1) for k,v in x.items()} if depth < 2 else list(x)[:20]
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}[{len(x)}]"
    if hasattr(x, "shape"):
        return f"{type(x).__name__}{tuple(x.shape)}"
    return repr(x)[:120]

for _name in ["dose_directions", "head_models", "state_models", "surrogates",
              "response_models", "probe_ids", "validation_ids", "state_data"]:
    if _name in globals():
        print(_name, compact_structure(globals()[_name]))

print("=== VOCABULARY KEYS AND NORMALIZATION ===")
print("keys:", list(VOCAB))
for _key in VOCAB:
    if any(s in _key.lower() for s in ["norm", "code", "config"]):
        print(_key, str(VOCAB[_key])[:1800])

print("=== RELEVANT TENSORS ===")
for _name in NAMES:
    if (_name.startswith("encode_action") or _name.startswith("core.c_out") or
        _name.startswith("core.out_norm") or _name.startswith("attr_heads.value") or
        _name.startswith("z_post") or _name.startswith("core.z_post")):
        print(_name, SHAPES[_name])

print("=== FORWARDS 6–7 FILES ===")
for _k in [6, 7]:
    _folder = f"/work/fwd_{_k}"
    print(_folder, os.listdir(_folder))
    for _file in os.listdir(_folder):
        if _file.endswith(".csv"):
            _df = pd.read_csv(_folder + "/" + _file)
            print(_file, list(_df.columns), _df.shape)
            if "nll" in _df:
                print("mean NLL:", _df["nll"].mean())
            if "edit" in _df:
                _num = [c for c in _df.select_dtypes(include="number").columns
                        if c not in ["stay_id", "hour", "stay"]]
                print(_df.groupby("edit")[_num].mean().to_string())

print("=== STATE FILES ===")
for _k in [3, 8]:
    _folder = f"/work/fwd_{_k}"
    for _file in os.listdir(_folder):
        if _file.endswith(".npz"):
            _z = np.load(_folder + "/" + _file)
            print(_folder, _file, {k: _z[k].shape for k in _z.files})
```
