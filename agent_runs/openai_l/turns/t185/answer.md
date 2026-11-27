The full-record test passed: value NLL rose by only about 0.00006 nats, far below the 0.005 limit. This supports keeping the quiet write provisionally, but says nothing yet about causal accuracy. I’ll inspect the decoder and stored samples to construct a readout that responds to changed doses while remaining small on factual records.

```python
import numpy as np
import dis
from pathlib import Path

print("=== VALUE-HEAD TENSORS ===")
for name in NAMES:
    if any(s in name for s in ["attr_heads.value", "last_numeric", "query_norm",
                               "z_head", "z_prior", "core.out_norm"]):
        print(name, SHAPES[name])

print("\n=== LOCAL HELPER SIGNATURES ===")
for name in sorted(list(globals())):
    obj = globals()[name]
    if callable(obj) and hasattr(obj, "__code__"):
        if any(ch.isdigit() for ch in name) and not name.startswith("_"):
            print(name, obj.__code__.co_varnames[:obj.__code__.co_argcount])

print("\n=== DECODER INTERFACES ===")
for name in ["decoder_capture96", "head_inputs96", "prelude_features102"]:
    fn = globals().get(name)
    if fn is not None:
        print(name, "globals", fn.__code__.co_names,
              "constants", repr(fn.__code__.co_consts)[:1600])
        if name != "prelude_features102":
            dis.dis(fn)

print("\n=== STORED SAMPLE / FEATURE OBJECTS ===")
for name in sorted(list(globals())):
    obj = globals()[name]
    if hasattr(obj, "shape") and not name.startswith("_"):
        if any(s in name.lower() for s in [
            "ctx", "context", "sample", "hidden", "dose", "factual",
            "feature", "pre123", "125", "152", "167", "96"
        ]):
            print(name, tuple(obj.shape))

print("\n=== NOTEBOOK SECTION TITLES ===")
for section in Path("/work/notes.md").read_text().split("\n## ")[1:]:
    print(section.split("\n", 1)[0])

with open("/work/notes.md", "a") as f:
    f.write("\n## Forward 8 / quiet write factual audit\n"
            "100 tuning stays, 4751 readings: original NLL 1.63164, edited "
            "1.63170 (delta about +0.00006). Quiet b_in write accepted "
            "provisionally. No causal value readout exists yet; next recover "
            "decoder features and fit a factual-anchored dose correction.\n")
```