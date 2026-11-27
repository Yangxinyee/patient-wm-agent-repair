The architecture confirms that feedback from generated readings matters, so a one-step correction is insufficient. The notebook also shows that early causal estimates were rejected; I won’t use those targets. Since the recovery output was truncated, I’ll now retrieve only the later evidence and recent components. These must support a dose-generalizing correction, not merely a good fit to the original records.

```python
import re, inspect, numpy as np, torch

parts127 = re.split(r"(?m)^## (.+)\n", open("/work/notes.md").read())
sections127 = list(zip(parts127[1::2], parts127[2::2]))
causal127 = []
for heading, body in sections127:
    m = re.search(r"run\s+(\d+)", heading, re.I)
    if m and 12 <= int(m.group(1)) <= 80:
        if re.search(r"causal|identif|threshold|effect|target|dynamics|instrument", body, re.I):
            causal127.append((heading, body))
print("=== LATER RECORD EVIDENCE ===")
for heading, body in causal127[-12:]:
    print("\n##", heading)
    print(body.strip()[:430])

g127 = dict(globals())
print("\n=== CAUSAL/TARGET VARIABLE INVENTORY ===")
for k, v in sorted(g127.items()):
    if re.search(r"caus|target|beta|rho|phys|truth|effect|dyn", k, re.I):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, tuple(v.shape),
                  v.tolist() if v.numel() <= 16 else "" if isinstance(v, torch.Tensor)
                  else "")
        elif isinstance(v, (int, float, str)):
            print(k, repr(v)[:200])
        elif isinstance(v, dict):
            print(k, "dict keys:", list(v)[:15])
        else:
            print(k, type(v).__name__)

print("\n=== FUNCTION SIGNATURES ===")
for k, v in sorted(g127.items()):
    if inspect.isfunction(v) and not k.startswith("_"):
        try:
            print(k, inspect.signature(v))
        except Exception:
            print(k)

print("\n=== RUN-125 COMPONENTS ===")
for k, v in sorted(g127.items()):
    if k.endswith("125"):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, {str(x): getattr(y, "shape", type(y).__name__)
                      for x, y in list(v.items())[:12]})
        elif isinstance(v, (int, float, str, tuple, list)):
            print(k, repr(v)[:250])
        else:
            print(k, type(v).__name__)
```
