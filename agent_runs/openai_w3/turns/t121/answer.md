The diagnostic stopped on a Boolean array; no weights changed. The dose decoder and latent-token gate are available. I’ll retrieve only the missing calibration objects next, then test a small latent innovation channel. It must preserve factual NLL before I trust its intervention response.

```python
import re, numpy as np, pandas as pd, torch

print("=== RECENT OBJECT SHAPES ===")
for _name110 in sorted(list(globals())):
    if not re.search(r"(103|104|105)$", _name110):
        continue
    _obj110 = globals()[_name110]
    if isinstance(_obj110, (np.ndarray, torch.Tensor)):
        print(_name110, tuple(_obj110.shape))
    elif isinstance(_obj110, dict):
        print(_name110, "keys", list(_obj110)[:8])
    elif isinstance(_obj110, (list, tuple)):
        print(_name110, "length", len(_obj110))

print("\n=== CORE IMPLEMENTATION CONSTANTS ===")
for _name110 in ("state_update82", "memory82", "layer82", "means104"):
    _fun110 = globals()[_name110]
    print(_name110)
    print("references", _fun110.__code__.co_names)
    print("constants", repr(_fun110.__code__.co_consts)[:950])

print("\n=== FACTUAL REFERENCE ===")
_path110 = next(iter(sorted(Path("/work").glob("fwd_*/factual_original.csv"))))
_df110 = pd.read_csv(_path110)
print(_path110, _df110.shape)
print(_df110.head(3).to_string(index=False))

print("\n=== SELECTED VALUES ===")
for _name110 in ("policy_fits103", "decoder103", "gate103", "constraints104",
                 "readout_tests97", "means_replay104"):
    if _name110 not in globals():
        continue
    _obj110 = globals()[_name110]
    if isinstance(_obj110, (np.ndarray, torch.Tensor)):
        _arr110 = as_numpy(_obj110).astype(float)
        print(_name110, _arr110.shape,
              "sample", np.round(_arr110.reshape(-1)[:12], 5))
    else:
        print(_name110, repr(_obj110)[:1200])

print("\n=== TARGET TENSORS ===")
for _name110 in NAMES:
    if any(_s110 in _name110 for _s110 in
           ("type_emb", "last_", "attr_heads.value", "core.prelude.0.attn",
            "z_post", "z_in")):
        print(_name110, SHAPES[_name110])
print("CURRENT EDITS", edited())
```
