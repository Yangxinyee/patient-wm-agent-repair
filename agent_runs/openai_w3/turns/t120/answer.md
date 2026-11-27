The local readout uses type embedding 1 and a cached register contribution. Either could explain the large error. I’ll check those against the vocabulary and factual rows, using compact diagnostics to avoid another truncation. Matching event identities and means is the next validation gate.

```python
import re, numpy as np, pandas as pd, torch

print("=== VOCABULARY ===")
print("codes", VOCAB["codes"])
print("config", VOCAB.get("config"))
print("top-level keys", list(VOCAB))
print("README", Path("/data/README.md").read_text()[:3500])

print("\n=== READOUT DATA ===")
for _name109 in sorted(list(globals())):
    if not re.search(r"(105|104|103|97|90)$", _name109):
        continue
    _obj109 = globals()[_name109]
    if isinstance(_obj109, (np.ndarray, torch.Tensor)):
        _arr109 = as_numpy(_obj109)
        print(_name109, _arr109.shape,
              "first", np.round(_arr109.reshape(-1)[:6], 4))
    elif isinstance(_obj109, dict):
        print(_name109, "dict keys", list(_obj109)[:12])
    elif isinstance(_obj109, (list, tuple)):
        print(_name109, len(_obj109), repr(_obj109[:3])[:160])

print("\n=== UPDATE REFERENCES AND CONSTANTS ===")
for _name109 in ("state_update82", "memory82", "layer82", "means104", "means105"):
    _fun109 = globals()[_name109]
    print(_name109, "names", _fun109.__code__.co_names,
          "constants", repr(_fun109.__code__.co_consts)[:1100])

print("\n=== FACTUAL EXAMPLE ===")
_path109 = next(iter(sorted(Path("/work").glob("fwd_*/factual_original.csv"))))
_df109 = pd.read_csv(_path109)
print(_path109, _df109.shape)
print(_df109.head(5).to_string(index=False))

print("\n=== RELEVANT TENSOR SHAPES ===")
for _name109 in NAMES:
    if any(_s109 in _name109 for _s109 in
           ("type_emb", "last_", "attr_heads.value", "embed_transition.value",
            "core.prelude.0.ff", "core.prelude.0.attn", "z_post", "z_in")):
        print(_name109, SHAPES[_name109])
print("CURRENT EDITS", edited())
```
