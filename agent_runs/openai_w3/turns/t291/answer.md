The retained edit combines a large attention payload with an FFN compensation fitted on factual states. Earlier tests also found that one-step dose responses had the right signs, but longer rollouts became asymmetric. I’ll inspect the exact decomposition so the next fit can constrain the total dose response—not just one component. Small factual residuals and linear, correctly signed dose contrasts would support this approach.

```python
import dis, io

print("=== TOTAL-PRELUDE HELPER ===")
buf258 = io.StringIO()
dis.dis(parts254, file=buf258)
print(buf258.getvalue()[:4300])

print("\n=== CALIBRATION TARGETS ===")
print("target255:", target255)
print("J255 row norms:", np.linalg.norm(J255, axis=1))
print("directions255 norms:", np.linalg.norm(directions255, axis=0))
print("primary/full norms:",
      np.linalg.norm(directions_primary255, axis=0),
      np.linalg.norm(directions_full255, axis=0))

print("\n=== TOKEN / CONTRAST / CALIBRATION CACHES ===")
for k in sorted(list(globals())):
    if k.startswith("_"):
        continue
    if (any(s in k.lower() for s in ["token", "contrast", "calib"]) or
        k.endswith(("185", "206", "219", "243", "250"))):
        v = globals()[k]
        if callable(v):
            print(k, "FUNCTION", str(inspect.signature(v)))
        elif isinstance(v, dict):
            print(k, "DICT", [(str(a), tuple(b.shape) if hasattr(b, "shape") else type(b).__name__)
                             for a, b in list(v.items())[:9]])
        elif hasattr(v, "shape"):
            print(k, type(v).__name__, tuple(v.shape))
        else:
            print(k, type(v).__name__, str(v)[:80])

print("\n=== VALUE HEAD TENSOR SHAPES ===")
for n in NAMES:
    if any(s in n for s in ["attr_heads.value", "last_numeric", "z_head", "core.out_norm"]):
        print(n, SHAPES[n])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 258; forwards 20. Current FFN compensation penalized only its own dose contrasts, not the total edited prelude response. Inspecting decomposition and cached changed-dose tokens for a total-response constrained regression. No weights changed.\n")
```