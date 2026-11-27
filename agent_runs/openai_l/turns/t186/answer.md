The value head is a two-layer network, and validated CPU helpers already exist for its inputs and one-step transitions. I’ll recover their exact interfaces rather than rebuild them. A useful correction must preserve factual logits while retaining dose contrasts; failure on either criterion would rule out that readout.

```python
import dis
import pandas as pd
import numpy as np
from pathlib import Path

def interface171(name, max_lines=16):
    fn = globals().get(name)
    if fn is None or not hasattr(fn, "__code__"):
        return
    print("\n", name, "ARGS:", fn.__code__.co_varnames[:fn.__code__.co_argcount])
    print("GLOBALS:", fn.__code__.co_names)
    print("CONSTANTS:", repr(fn.__code__.co_consts)[:900])
    lines = {}
    line = None
    for ins in dis.get_instructions(fn):
        if ins.starts_line is not None:
            line = ins.starts_line
        if ins.opname not in {"RESUME", "CACHE", "PRECALL", "COPY", "SWAP", "NOP"}:
            lines.setdefault(line, []).append(ins.opname + ":" + ins.argrepr)
    for lineno, items in list(lines.items())[:max_lines]:
        print(lineno, " ".join(items)[:500])

for name in ["w63", "decode67", "make_cf81", "transition64",
             "context64", "prepare103", "bundle_inputs74"]:
    interface171(name, 10)

print("\n=== READING METADATA ===")
for name in sorted(list(globals())):
    obj = globals()[name]
    if isinstance(obj, pd.DataFrame) and any(
        s in name for s in ["65", "67", "73", "74", "80", "81", "102"]
    ):
        print(name, obj.shape, list(obj.columns)[:14],
              obj.iloc[:1].to_dict("records"))

print("\n=== STATE FILES ===")
for path in sorted(Path("/work").glob("fwd_*/*.npz")):
    print(str(path), round(path.stat().st_size / 1e6, 2), "MB")

print("\n=== KERNEL / FIT OBJECTS ===")
for name in sorted(list(globals())):
    if any(s in name.lower() for s in ["kernel", "pole", "lagfit", "kernel_fit"]):
        obj = globals()[name]
        if hasattr(obj, "shape"):
            print(name, tuple(obj.shape), np.asarray(obj).reshape(-1)[:16])
        elif isinstance(obj, dict):
            print(name, list(obj)[:16])
        elif isinstance(obj, (list, tuple)):
            print(name, repr(obj)[:600])

print("\n=== CAUSAL NOTE TITLES ===")
sections171 = Path("/work/notes.md").read_text().split("\n## ")
for section in sections171:
    if any(s in section.lower() for s in [
        "lag kernel", "eight-lag", "eight lag", "distributed", "accepted immediate"
    ]):
        print(section[:1700])

fit_stays171 = pd.read_csv("/data/train.csv", usecols=["stay_id"])["stay_id"].drop_duplicates().tolist()[:100]
print("\nTRAINING STATE-FIT STAYS:", fit_stays171)
```