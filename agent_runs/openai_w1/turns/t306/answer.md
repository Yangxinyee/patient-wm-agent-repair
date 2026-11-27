The architecture confirms that an injected state signal passes through the coda before returning to the dose detector, so immediate orthogonality is insufficient. I’ll recover that path and the record-based effect estimates. If the transformed feedback can be suppressed while preserving dose sensitivity, this route may still work; otherwise I’ll reject it.

```python
import dis, re
from pathlib import Path

print("=== COMPACT LOCAL FUNCTIONS ===")
for fn_name in ("transition214", "direct_readout257"):
    fn = globals()[fn_name]
    print("\n", fn_name, "constants:", fn.__code__.co_consts)
    ins = list(dis.get_instructions(fn))
    for start in range(0, len(ins), 12):
        print(" | ".join(f"{i.opname} {i.argrepr}" for i in ins[start:start+12]))

print("\n=== TOKEN / EXPOSURE CACHE ===")
for name in sorted(list(globals())):
    if (name.startswith(("ds_", "Es259", "factualE258", "nt215", "nt240", "base238",
                         "E258", "rx_target", "rx_pick", "stays248"))
        or name in ("finite_readout221", "prelude244", "delta_in", "target263")):
        obj = globals()[name]
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            print(name, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(name, "dict", [(str(k), getattr(v, "shape", type(v).__name__)) for k,v in list(obj.items())[:12]])
        elif callable(obj):
            print(name, "function", obj.__code__.co_varnames[:obj.__code__.co_argcount])
        else:
            print(name, repr(obj)[:300])

notes268 = Path("/work/notes.md").read_text()
print("\n=== EARLY RECORD IDENTIFICATION ===")
print(notes268[:4500])
print("\n=== SAFE236 NOTE ===")
pos268 = notes268.rfind("## Run 236")
print(notes268[pos268:pos268+1200] if pos268 >= 0 else "No separate run236 heading.")
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 268\nRecovered local transition/coda path, feature-cache alignment and earlier record-identification notes. No weights changed. Counts: 268 Python runs, 28 forwards.\n")
```
