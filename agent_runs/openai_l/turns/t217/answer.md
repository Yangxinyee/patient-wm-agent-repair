The inspection located the pre-activation features, dose directions, and sensor gate, but stopped when a display helper encountered event names. No weights changed. I’ll inspect those objects safely now; a reliable gate would support a nonlinear correction that affects only the appropriate sensor family.
```python
import inspect

def describe196(name, value):
    if isinstance(value, dict):
        print(name, "dict", list(value)[:12])
    elif hasattr(value, "columns"):
        print(name, value.shape, list(value.columns))
        print(value.head(2).to_string(index=False))
    elif hasattr(value, "shape"):
        a = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        if a.dtype.kind in "biufc":
            print(name, a.shape, "range", float(np.nanmin(a)), float(np.nanmax(a)))
        else:
            print(name, a.shape, "sample", a.reshape(-1)[:8].tolist())
    else:
        print(name, type(value).__name__, repr(value)[:160])

print("=== DATA AND FEATURES ===")
for n in ["train", "tuning", "events74", "events179", "centres_train67",
          "eps67", "base180", "Yfact179", "YF179", "gate192", "gatepred192",
          "gate_y192", "quiet192", "importance192", "fitmask179"]:
    if n in globals():
        describe196(n, globals()[n])

print("\n=== COUNTERFACTUAL DICTIONARIES ===")
for n in sorted(list(globals())):
    v = globals()[n]
    if isinstance(v, dict) and v and any(
        isinstance(k, tuple) and k and k[0] in ("insulin", "norepinephrine")
        for k in v):
        print(n, "keys", list(v)[:12])
for k in [("insulin", .1), ("norepinephrine", .1)]:
    describe196(str(k), directions193[k])
describe196("base180 rows", base180["rows"])
describe196("base180 delta_rate", base180["delta_rate"])

print("\n=== GATE QUALITY ===")
print("factual gate error RMS",
      float(np.sqrt(np.mean((gatepred192 - gate_y192)**2))))
for ev in np.unique(events179):
    m = events179 == ev
    g = gatepred192[:, m].reshape(-1)
    print(ev, "gate quantiles", np.quantile(g, [0,.01,.5,.99,1]).round(4).tolist())

print("\n=== HELPER DEFINITIONS ===")
for n in ["target_logits83", "nll179", "head_inputs96", "decoder_base180"]:
    f = globals()[n]
    print("\n", n)
    try:
        print(inspect.getsource(f))
    except (OSError, TypeError):
        c = f.__code__
        print("args", c.co_varnames[:c.co_argcount],
              "locals", c.co_varnames,
              "names", c.co_names,
              "constants", tuple(x if isinstance(x, (str, int, float, tuple, type(None))) else type(x).__name__
                                 for x in c.co_consts))
        if n == "target_logits83":
            dis.dis(f)

print("\n=== EARLY RECORD-EFFECT NOTES ===")
print(Path("/work/notes.md").read_text()[:6500])
```
