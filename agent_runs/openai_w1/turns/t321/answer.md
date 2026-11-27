The architecture confirms that doses enter through the posterior and prelude before the recurrent update. The previous output was truncated by a function’s large default argument, so the needed helpers remain unclear. I’ll inspect them compactly and rebuild the incoming state from the new export. Exact record alignment is required before fitting anything.

```python
import torch, inspect, re
from pathlib import Path
torch.set_num_threads(8)

print("=== RECORD-IDENTIFICATION NOTES ===")
print(Path("/work/notes.md").read_text()[:4200])

g281 = dict(globals())
fn281 = {
    n: o for n, o in g281.items()
    if inspect.isfunction(o) and getattr(o, "__module__", "") == "__main__"
}
print("\n=== LOCAL FUNCTION NAMES ===")
print(", ".join(sorted(fn281)))

print("\n=== COMPACT RECENT OBJECTS ===")
for n in sorted(g281):
    if not re.search(r"27[6789]|safe236|ds_rates", n):
        continue
    o = g281[n]
    if inspect.isfunction(o):
        c = o.__code__
        print(n, "args", c.co_varnames[:c.co_argcount],
              "uses", c.co_names)
    elif hasattr(o, "shape"):
        print(n, tuple(o.shape))
    elif isinstance(o, dict):
        print(n, "dict", len(o), "keys", list(o)[:8])
    elif isinstance(o, (list, tuple)):
        print(n, type(o).__name__, len(o))
    elif not isinstance(o, str):
        print(n, repr(o)[:100])

print("\n=== RECONSTRUCTION HELPER SOURCES ===")
for n, fn in sorted(fn281.items()):
    if n.endswith("277") or ("prelude" in n.lower()) or ("replay" in n.lower()):
        try:
            src = inspect.getsource(fn)
            print(n, "\n", src[:3200])
        except (OSError, TypeError):
            c = fn.__code__
            print(n, "args", c.co_varnames[:c.co_argcount],
                  "uses", c.co_names,
                  "scalar constants",
                  [x for x in c.co_consts if isinstance(x, (int, float, str))][:20])

print("\n=== STATE-EXPORT ALIGNMENT ===")
for n, o in sorted(g281.items()):
    if isinstance(o, dict) and {"x", "stay_id", "hour"}.issubset(o):
        try:
            if np.shape(o["x"]) == (5215, 992):
                print(n,
                      "same stays", np.array_equal(o["stay_id"], S280["stay_id"]),
                      "same hours", np.allclose(o["hour"], S280["hour"], atol=1e-8))
        except Exception:
            pass

n281 = len(S280["stay_id"])
same281 = np.r_[False, S280["stay_id"][1:] == S280["stay_id"][:-1]]
assert tuple(hflow277.shape) == (n281, 4, 192)
assert np.shape(dt277) == (n281,)
Hflow281 = hflow277.detach().clone()
Oprev281 = torch.zeros((n281, 192), dtype=torch.float32)
ix281 = np.flatnonzero(same281)
flow_a281 = torch.as_tensor(load("core.flow.a"), dtype=torch.float32)
decay281 = torch.exp(
    -torch.as_tensor(dt277, dtype=torch.float32)[ix281, None, None]
    * torch.exp(flow_a281)
)
Hflow281[ix281] = torch.from_numpy(
    S280["x"][ix281 - 1, 192:960].reshape(-1, 4, 192)
) * decay281
Oprev281[ix281] = torch.from_numpy(S280["x"][ix281 - 1, :192])
print("Rebuilt flowed incoming state:", tuple(Hflow281.shape))
print("RMS change versus original incoming state:",
      float(torch.sqrt(torch.mean((Hflow281 - hflow277)**2))))
print("Stay starts:", int((~same281).sum()),
      "; their initial flow is retained from the verified reconstruction.")

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 281\nRun280 output was truncated by verbose function defaults. Compact helper recovery and export alignment audit; rebuilt current incoming h for noninitial bundles using exported previous h and verified rounded-minute dt. No weights changed. Counts: 281 Python runs, 30 forwards.\n")
```
