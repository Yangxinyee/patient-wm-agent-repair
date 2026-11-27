The later record analyses agree on an immediate insulin effect near −1.95 and a norepinephrine effect near +3.1, with different delayed responses. Insulin also has independent threshold evidence; norepinephrine relies on adjusted record comparisons. I’ll recover the exact response kernels and recurrence interfaces next. This will let me test a correction across doses and time, rather than tune only the hour-12 doubled-dose readout.

```python
import dis, inspect, numpy as np, torch

print("=== ACCEPTED KERNELS ===")
print("kernels19 =", kernels19)
if "ar2_predictions19" in globals():
    print("ar2_predictions19 =", repr(ar2_predictions19)[:1800])

print("\n=== REMAINING HELPER SIGNATURES ===")
for k, v in sorted(dict(globals()).items()):
    if k >= "pool_actions49" and inspect.isfunction(v) and not k.startswith("_"):
        try:
            print(k, inspect.signature(v))
        except Exception:
            print(k)

def brief_dis129(f, limit=1600):
    rows = [f"{i.opname} {i.argrepr}" for i in dis.get_instructions(f)]
    text = "\n".join(rows)
    print(f.__name__, "globals:", f.__code__.co_names)
    print(text[:limit])

print("\n=== WEIGHT ACCESS ===")
brief_dis129(linear63, 700)
print("\n=== ATTENTION FEATURE EXTRACTION ===")
brief_dis129(attention_features116, 2400)

print("\n=== CONTROLLER STRUCTURE ===")
for k, v in controllers125.items():
    print("penalty", k, "type", type(v).__name__)
    if isinstance(v, dict):
        print({str(a): getattr(b, "shape", type(b).__name__) for a, b in v.items()})
    elif isinstance(v, (tuple, list)):
        print([(type(a).__name__, getattr(a, "shape", None)) for a in v])
    else:
        print(getattr(v, "shape", None))

for a in ("P125", "T125", "QE125", "SP125"):
    x = globals()[a]
    x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    print(a, "column norms", np.round(np.linalg.norm(x, axis=0), 5))
    for b in ("P125", "T125", "QE125", "SP125"):
        if a < b:
            y = globals()[b]
            y = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
            print(a + ".T @ " + b, np.round(x.T @ y, 4))

print("\n=== RELEVANT RETAINED VARIABLES ===")
names129 = sorted(k for k in globals()
                  if re.search(r"(81|96|97|103|118|119|123|124)$", k))
print(" ".join(names129))
```
