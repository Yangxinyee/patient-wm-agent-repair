This repeats the previous rollout result, so there is no new evidence. The stronger edit remains factually safe on these stays but causally asymmetric. I’ll recover the saved carrier implementation and checkpoints; that should let the next experiment target the asymmetry rather than repeat the same failed constraint.

```python
import types, inspect, re
checkpoint287 = {}
for n in edited():
    x = load(n)
    checkpoint287[n] = x.clone() if hasattr(x, "clone") else np.array(x, copy=True)

print("CANDIDATE STRUCTURES")
for label, obj in [("candidate285", candidate285), ("candidates267", candidates267),
                   ("instructions277", globals().get("instructions277")),
                   ("instr", globals().get("instr"))]:
    print(label, type(obj).__name__)
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:8]:
            print(" ", k, type(v).__name__,
                  list(v.keys()) if isinstance(v, dict) else getattr(v, "shape", None))
    elif isinstance(obj, str):
        print(obj[:1000])

print("\nRELEVANT VARIABLES")
print([k for k in list(globals()) if re.search(
    r"carrier|replay|rho|beta|theta|coeff|phys|target|direc|effect|r233|post|build|token", k, re.I)])

print("\nKERNEL FUNCTIONS")
for k, v in list(globals().items()):
    if isinstance(v, types.FunctionType) and hasattr(v, "__code__"):
        if not v.__code__.co_filename.startswith(("/usr/", "/opt/", "/home/")):
            print(k, str(inspect.signature(v)))

print("\nEDIT NORMS")
for n in edited():
    x, o = load(n), load_original(n)
    x = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    o = o.detach().cpu().numpy() if hasattr(o, "detach") else np.asarray(o)
    print(n, "original", round(float(np.linalg.norm(o)), 3),
          "edit", round(float(np.linalg.norm(x-o)), 3))

print("\nEARLY IDENTIFICATION NOTES")
print(open("/work/notes.md").read()[:3500])
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 287; forwards 28. Gamma=10 passes factual (+.00049 nats), but rollout shifts (-19.716,+1.096,+2.232,-.820) remain asymmetric. Saved as checkpoint287. Recovering carrier implementation; no new weights installed.\n")
```