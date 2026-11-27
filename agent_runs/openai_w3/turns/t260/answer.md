The record estimates show approximately linear immediate insulin effects and a longer-lived glucose response. I will now check the transport magnitudes and readout construction. If those directions are much larger than ordinary model states, their local calibration may fail during repeated rollout.

```python
import dis

print("PRELUDE INSERTION AND NORMALIZATION")
r232_ins = list(dis.Bytecode(prelude155))
r232_start = next(i for i, x in enumerate(r232_ins)
                  if x.opname == "LOAD_FAST" and x.argval == "injection")
print(" | ".join(f"{x.opname} {x.argrepr}" for x in r232_ins[r232_start:])[:2700])

for name in ["view171", "phys171"]:
    f = globals()[name]
    print("\n", name, "locals", f.__code__.co_varnames)
    print(" | ".join(f"{x.opname} {x.argrepr}" for x in dis.Bytecode(f))[:1800])

print("\nTRANSPORT SCALE")
for name in ["directions186", "dirs_full228", "J186", "s186", "base186"]:
    a = as_numpy(globals()[name])
    print(name, a.shape, "norm", np.linalg.norm(a, axis=0) if a.ndim == 2 else a)
print("ordinary slot-token RMS", float(np.sqrt(np.mean(as_numpy(tokens_orig169[:, :4])**2))))
print("ordinary E RMS", float(np.sqrt(np.mean(as_numpy(
    prelude155(tokens_orig169, overrides=orig_overrides228)[0])**2))))

print("\nCURRENT FIT OBJECTS")
for n, v in list(globals().items()):
    if n.endswith("220") and not inspect.isfunction(v):
        if hasattr(v, "shape"):
            print(n, tuple(v.shape))
        elif isinstance(v, (float, int, str)):
            print(n, repr(v)[:80])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 232; forwards 16. Checking whether transport directions exceed ordinary state scale and whether local readout calibration includes actual sensor registers. No edits changed.\n")
```