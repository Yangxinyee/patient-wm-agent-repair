The reached-hour summaries agree with the notebook: the insulin baseline gives about −12.7/+4.3, while its norepinephrine response remains too small. Rather than immediately weaken the insulin repair, I’ll test whether a closed-form, state-dependent decoder correction can recover factual fit. It is useful only if it improves held-out NLL without substantially undoing the dose responses.

```python
import re, numpy as np, torch
from pathlib import Path

print("RECENT CALIBRATION ARRAYS")
recent181 = []
for k181 in sorted(list(globals())):
    if re.search(r"(?:161|162|168|175|176)$", k181):
        o181 = globals()[k181]
        if hasattr(o181, "shape"):
            recent181.append((k181, tuple(o181.shape)))
        elif isinstance(o181, dict):
            recent181.append((k181, "dict", list(o181.keys())[:5]))
        elif callable(o181):
            recent181.append((k181, "function"))
print(recent181)

print("\nDECODER / STATE HELPER SIGNATURES")
for k181 in sorted(list(globals())):
    o181 = globals()[k181]
    if callable(o181) and hasattr(o181, "__code__") and re.search(
            r"(prob|decode|query|head|readout|cf|counter|prelude|core|coda)", k181, re.I):
        c181 = o181.__code__
        print(k181, c181.co_varnames[:c181.co_argcount],
              "referenced globals", c181.co_names[:18])

print("\nDECODER TENSORS")
for n181 in NAMES:
    if n181.startswith(("attr_heads.value_bin", "query_norm", "last_numeric",
                        "embed_transition.value_bin", "z_head", "type_emb")):
        print(n181, SHAPES[n181])

print("\nTRAINING-STATE ARCHIVE")
p181 = Path("/work/fwd_22")
print([x.name for x in p181.iterdir()])
for p182 in sorted(p181.glob("*.npz")):
    with np.load(p182) as a181:
        print(p182.name, [(k, a181[k].shape) for k in a181.files])

print("\nARRAY DICTIONARIES")
for k181 in ["W", "weights", "params", "head_weights", "train_original168",
             "train_edited168", "train_meta161", "meta161", "p_original175",
             "p_edited175", "train_ids161"]:
    if k181 in globals():
        o181 = globals()[k181]
        if isinstance(o181, dict):
            print(k181, "keys", list(o181.keys())[:15])
        elif hasattr(o181, "shape"):
            print(k181, tuple(o181.shape))
        else:
            print(k181, type(o181).__name__)

with Path("/work/notes.md").open("a") as f181:
    f181.write("\n## Run 181: prepare state-conditioned factual distillation\n"
               "Reached-hour audit confirms baseline insulin responses around -12.7/+4.3. "
               "Inspecting paired training decoder inputs for an auxiliary closed-form ridge "
               "correction to the final value-head matrix; no gradients or weight optimiser, "
               "and no edit installed yet. Python runs: 181; forwards: 22.\n")
```
