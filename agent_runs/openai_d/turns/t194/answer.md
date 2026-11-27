The adjusted record fits support a negative insulin effect and a positive NE effect, with delayed responses contributing to the larger rollout targets. I’ll inspect the NE direction’s fitting and normalization. If its factual changes are mostly decoding noise, refitting should be safer than amplifying the previous trial.

```python
import re, dis
from pathlib import Path

print("=== NE METHOD NOTES ===")
_s169 = re.split(r"(?m)(?=^## )", Path("/work/notes.md").read_text())
for _s in _s169:
    if re.match(r"## Run (?:123|124|125|126|127|129|131|132|136|137|138|139)\b", _s):
        print(_s[:530])

print("\n=== NE FEATURE DICTIONARIES ===")
for _ridge, _f in r138_features.items():
    print("ridge", _ridge, "fields", list(_f))
    for _k, _v in _f.items():
        if hasattr(_v, "shape"):
            print(" ", _k, _v.shape)
        elif isinstance(_v, (int, float, str)):
            print(" ", _k, _v)
    break

print("\n=== SMALL SCALARS / ARRAYS ===")
for _n in ["r138_alpha", "r138_tau", "r138_slots", "r139_jplus", "r139_jminus",
           "r139_scale", "r139_error", "r139_db", "r139_gb",
           "r138_tilt", "r139_tilt", "r132_dmean", "r132_gb",
           "r132_response_fit", "r132_response_val"]:
    _v = globals().get(_n)
    print(_n, repr(_v)[:600])

print("\n=== LOCAL DATA AND MATRIX SHAPES ===")
for _n in ["r137_Xnp", "r137_probe_features", "r136_transition",
           "r138_G", "r138_Gprobe", "r138_removed", "r138_L0",
           "r144_D", "r144_G", "r145_X", "r145_transition",
           "r132_trainX", "r132_feat", "r132_family",
           "r132_fit", "r132_val", "head_meta", "rec_a"]:
    _v = globals().get(_n)
    print(_n, type(_v).__name__, getattr(_v, "shape", ""),
          list(_v.columns)[:18] if hasattr(_v, "columns") else "")
print("fit/validation readings:", int(r137_fit.sum()), int(r137_val.sum()))
print("head_meta stays:", head_meta.stay_id.min(), head_meta.stay_id.max())

print("\n=== PAIR BYTECODE ===")
print(dis.Bytecode(r138_pair).dis()[:2200])
print(dis.Bytecode(r159_pair).dis()[:1500])
```
