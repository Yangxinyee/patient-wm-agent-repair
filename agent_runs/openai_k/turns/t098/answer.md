The architecture confirms that the latent-token route changes shared state, so it can affect every prediction head. The inspection output was truncated before reaching the candidate details. I’ll retrieve those details in a smaller output; this will distinguish a miscalibrated correction from a route that cannot isolate the drug effects.

```python
import inspect, numpy as np, torch

def brief92(x):
    if isinstance(x, (np.ndarray, torch.Tensor)):
        return f"{type(x).__name__} shape={tuple(x.shape)}"
    if isinstance(x, dict):
        return f"dict[{len(x)}] keys={list(x)[:12]}"
    if isinstance(x, (list, tuple)):
        parts = []
        for v in x[:4]:
            if isinstance(v, (np.ndarray, torch.Tensor)):
                parts.append(str(tuple(v.shape)))
            elif isinstance(v, dict):
                parts.append(f"dict[{len(v)}]")
            else:
                parts.append(repr(v)[:45])
        return f"{type(x).__name__}[{len(x)}] {parts}"
    if callable(x):
        try:
            return "function " + str(inspect.signature(x))
        except Exception:
            return "callable"
    return repr(x)[:150]

recent92 = sorted(k for k in list(globals())
                  if not k.startswith("_") and k.endswith(("88", "89", "90")))
print("RECENT OBJECTS")
for k in recent92:
    print(k, brief92(globals()[k]))

print("\nRELEVANT HELPERS")
helpers92 = []
for k in sorted(list(globals())):
    x = globals()[k]
    if inspect.isfunction(x) and any(s in k.lower() for s in
                                    ("core", "prior", "token", "head", "carrier")):
        helpers92.append(k)
for k in helpers92[:35]:
    print(k, brief92(globals()[k]))

print("\nCANDIDATE DETAILS")
for k in ("diagnostics90", "candidates90", "D88", "J88", "D88_candidates"):
    if k not in globals():
        continue
    x = globals()[k]
    print(k, brief92(x))
    if isinstance(x, dict):
        for a, b in list(x.items())[:3]:
            print(" ", a, brief92(b))
            if isinstance(b, dict):
                for c, d in list(b.items())[:8]:
                    print("   ", c, brief92(d))
    elif k == "diagnostics90":
        print(repr(x)[:1800])

print("\nSubmitted edits:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 92: compact candidate recovery\n"
            "- Architecture read successfully; shared latent-token edits can alter all heads.\n"
            "- Run 91 output truncated before candidate interfaces; retrieving them compactly.\n"
            "- No weights edited. Python runs: 92; forward passes: 4.\n")
```