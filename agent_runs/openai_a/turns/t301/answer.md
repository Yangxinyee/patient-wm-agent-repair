The saved checkpoints are intact, including the full-400 factual-tested fallback. The helpers also show that the edit combines a dose contrast with reading-type and insulin-activity gates. I’ll inspect that construction and the causal evidence directly. A calibration trial is worth pursuing only if it can reduce factual error without removing the dose contrast.

```python
import dis, inspect, re

print("=== EXACT CURRENT GRAFT CONFIGURATION ===")
for k, v in activity_gate_config.items():
    if isinstance(v, np.ndarray):
        print(k, "shape", v.shape, "norm", float(np.linalg.norm(v)))
    else:
        print(k, v)
print("graft rows:", activity_graft_rows)

print("\n=== GRAFT HELPER ===")
f = make_selective_contrast_graft
print(inspect.signature(f))
for ins in dis.get_instructions(f):
    print(ins.opname, ins.argrepr)

print("\n=== LOSS / PROBABILITY HELPER SIGNATURES ===")
for n in ("cache_losses", "cpu_head_parts", "cpu_head_probs", "graft_probs",
          "screen_value_candidate", "show_function", "matched_causal_targets163",
          "ne_threshold_rd", "make_ne_data"):
    f = globals()[n]
    print(n, inspect.signature(f), "names:", f.__code__.co_names[:18])

print("\n=== STAY-ID LISTS ===")
for n, v in list(globals().items()):
    if "id" in n.lower() and isinstance(v, (list, np.ndarray)) and not n.startswith("_"):
        if len(v) and np.ndim(v) == 1:
            print(n, "length", len(v), "first", list(v[:3]))

print("\n=== FOCUSED CAUSAL NOTES ===")
secs = re.split(r"(?m)(?=^## )", Path("/work/notes.md").read_text())
focus = [s for s in secs if any(w in s.splitlines()[0].lower()
         for w in ("causal", "instrument", "persistence", "discontinuity",
                   "insulin rd", "ne identification"))]
for s in focus[:6]:
    print(s[:600])
```
