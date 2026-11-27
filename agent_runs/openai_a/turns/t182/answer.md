The retained graft adds a gated scalar correction to the value logits; its stronger version did not reliably strengthen the insulin response. I’ll inspect its dose sensitivity and history protections. A useful replacement should respond to dose changes while remaining nearly insensitive to reading-only changes.

```python
import dis

print("=== CURRENT GRAFT CONSTRUCTION ===")
dis.dis(make_insulin_graft)
print("\n=== GRAFT PARAMETERS ===")
for n in ("graft_nodes", "graft_alpha", "graft_gate_scale",
          "sc_nodes", "sc_rank", "sc_old_scale"):
    print(n, globals().get(n))

print("\n=== CACHED DATA STRUCTURES ===")
def compact_structure(obj, depth=0):
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        return (type(obj).__name__, tuple(obj.shape))
    if isinstance(obj, pd.DataFrame):
        return ("DataFrame", obj.shape, list(obj.columns)[:14])
    if isinstance(obj, dict):
        return {str(k): compact_structure(v, depth+1) for k,v in list(obj.items())[:8]} if depth < 2 else list(obj)[:12]
    if isinstance(obj, (list, tuple)):
        return (type(obj).__name__, len(obj), [compact_structure(v, depth+1) for v in obj[:3]]) if depth < 2 else (type(obj).__name__, len(obj))
    return repr(obj)[:120]

for n in ("sc_fact", "sc_meta", "sc_sf", "sc_syn", "sc_active",
          "sc_dx", "sc_y", "sc_mult", "repair_train", "repair_tune",
          "rm_states", "rec_e_all", "rec_u_all", "ne_dml_results"):
    if n in globals():
        print(n, compact_structure(globals()[n]))

print("\n=== LOCAL-STEP HELPERS ===")
for n in sorted(list(globals())):
    o = globals()[n]
    if callable(o) and hasattr(o, "__code__") and any(
        s in n.lower() for s in ("rec_", "step", "transition", "synthetic", "posterior", "memory")
    ):
        print(n, o.__code__.co_varnames[:o.__code__.co_argcount],
              "uses", o.__code__.co_names[:20])

print("\n=== GRAFT DEVELOPMENT NOTES ===")
for s in sections:
    heading = s.splitlines()[0] if s.splitlines() else ""
    if any(t in heading.lower() for t in ("decoder", "protected", "innovation", "synthetic")):
        print(s[:700])
```
