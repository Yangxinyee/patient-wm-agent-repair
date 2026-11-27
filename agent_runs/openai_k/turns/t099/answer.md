The recovered candidates change both the posterior and its state-token projection. Their factual check is only local, while their dose responses appear to saturate. I’ll inspect the exact construction and dose units next. A clipping or scaling error would be repairable; otherwise this route needs redesign rather than a larger multiplier.

```python
import inspect, dis, numpy as np, torch

budget93 = 6500
for name93 in ("build_candidate89", "prior_features74", "r59_core",
               "hook_mlp90", "base_mlp90"):
    fn93 = globals().get(name93)
    if not inspect.isfunction(fn93):
        continue
    print("\nFUNCTION", name93, "REFERENCES", fn93.__code__.co_names)
    try:
        text93 = inspect.getsource(fn93)
    except Exception:
        text93 = dis.Bytecode(fn93).dis()
    take93 = min(len(text93), max(0, budget93))
    print(text93[:take93])
    budget93 -= take93
    if budget93 <= 0:
        break

print("\nTARGET AND DIRECTION CHECK")
for name93 in ("target88", "D89", "R88", "coef88", "Binnov89",
               "Bp89", "Bcomp89", "Btoken89", "delta88", "pred88"):
    if name93 not in globals():
        continue
    v93 = np.asarray(globals()[name93])
    print(name93, v93.shape, "range", float(np.min(v93)), float(np.max(v93)),
          "column norms", np.round(np.linalg.norm(v93, axis=0), 4).tolist()
          if v93.ndim == 2 and v93.shape[1] <= 4 else "")
    if name93 == "target88":
        print(v93)

print("\nCOUNTERFACTUAL FIELDS")
for container93 in ("CF89", "cf88"):
    if container93 not in globals():
        continue
    c93 = globals()[container93][0]
    for key93, val93 in c93.items():
        if isinstance(val93, (np.ndarray, torch.Tensor)):
            a93 = np.asarray(val93)
            print(container93, key93, a93.shape, a93.reshape(-1)[:8])
        else:
            print(container93, key93, repr(val93)[:150])

print("\nSTATE/HEAD FUNCTION NAMES")
print([k for k in sorted(list(globals()))
       if inspect.isfunction(globals()[k]) and
       any(s in k.lower() for s in ("r51", "r59", "80", "prob", "mean", "lin", "mlp"))])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 93: posterior clipping and unit audit\n"
            "- Inspecting candidate construction, response targets, and exact local core "
            "before another response calibration.\n"
            "- No weights edited. Python runs: 93; forward passes: 4.\n")
```