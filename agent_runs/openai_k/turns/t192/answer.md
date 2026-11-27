The normalization table is a dictionary keyed by `(event, attribute)`, not a function. The inspection stopped there, so it provided no new evidence about the edit. I’ll use the confirmed lookup and retrieve the saved candidate’s metrics and head features. This will show whether the correction can be tested with trustworthy inputs.

```python
from pathlib import Path
import re
import pandas as pd

print("CONFIRMED NORMALIZATION")
for event in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff", "insulin", "norepinephrine"):
    attribute = "value" if event.startswith(("glucose", "map")) else "rate"
    print(event, num_spec0[(event, attribute)])

print("RECENT INTERFACES")
r176_recent = []
for name, obj in list(globals().items()):
    if re.search(r"(166|170|171|172|173)$", name) or re.match(r"r17[0-3]_", name):
        if not name.startswith("__"):
            r176_recent.append((name, obj))
for name, obj in r176_recent[:65]:
    print(name, r175_desc(obj))

print("PREDICTION HELPERS")
for name in ("r59_features", "r59_core", "r49_readouts", "assay_head110",
             "lin0", "mlp0", "norm0", "update151", "local_outputs72"):
    obj = globals().get(name)
    if callable(obj) and hasattr(obj, "__code__"):
        print(name, obj.__code__.co_varnames[:obj.__code__.co_argcount],
              "uses", obj.__code__.co_names[:35])

print("HEAD SHAPES")
for name in NAMES:
    if any(s in name for s in ("attr_heads.value", "last_numeric", "query_norm", "z_head", "type_emb")):
        print(name, SHAPES[name])

print("FORWARD 10")
for p in sorted(Path("/work/fwd_10").iterdir()):
    print(p.name)
    if p.suffix in (".json", ".txt"):
        print(p.read_text()[:900])
    elif p.suffix == ".csv":
        q = pd.read_csv(p)
        print(q.shape, q.columns.tolist())
        print(q.head(2).to_string(index=False)[:600])

n = Path("/work/notes.md").read_text()
print("EARLY NOTES\n", n[:2000])
for tag in ("Run 139", "Run 140", "Run 171", "Run 172"):
    pos = n.rfind(tag)
    if pos >= 0:
        print("NOTE", n[pos:pos+700])

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 175–176: normalization lookup corrected\n"
            "- num_spec0 is a dictionary keyed by (event, attribute); run 175's "
            "disassembly attempt failed before candidate diagnostics.\n"
            "- Recovering compact interfaces and official forward-10 results.\n"
            "- Python runs: 176; forward passes: 10. Saved weights unchanged.\n")
```
