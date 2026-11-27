The feedback diagnostic explains the failure: the dose projections strongly read the same state directions they inject, creating a large feedback loop. I’ll constrain the projections against those directions and inspect how the coda transforms them. The approach remains viable only if those constraints reduce feedback without erasing the dose contrasts.

```python
import numpy as np
from scipy.linalg import null_space

C266 = delta265[:, None] * B257
Q266 = null_space(C266.T)
fits266 = {}
print("=== PROJECTIONS ORTHOGONAL TO IMMEDIATE INJECTION ===")
for lam in (0.1, 1.0, 10.0):
    psmall = np.linalg.solve(
        Q266.T @ (Gf263 + lam*Gd263 + ridge263*np.eye(192)) @ Q266,
        lam * Q266.T @ H263)
    P = Q266 @ psmall
    fits266[lam] = P
    pf = Ef263[hold263].reshape(-1, 192) @ P
    print("weight", lam,
          "factual RMS", np.round(np.sqrt(np.mean(pf**2, axis=0)), 4),
          "P norms", np.round(np.linalg.norm(P, axis=0), 3),
          "feedback max", np.max(np.abs(P.T @ C266)))
    for j, drug in enumerate(("insulin", "norepinephrine")):
        mask = hold263 & np.asarray(ds_s["has_" + drug], dtype=bool)
        result = []
        for mult in (0.0, 0.5, 1.5, 2.0, 3.0):
            de = np.asarray(Es259[(drug, mult)], dtype=np.float64) - Ef263
            p = de[mask].reshape(-1, 192) @ P
            target = np.repeat(ds_rates[mask, j]*(mult-1), 4)
            result.append((mult,
                           round(np.mean(p[:, j])/np.mean(target), 3),
                           round(np.mean(p[:, 1-j]), 3)))
        print(drug, "(multiplier, response/target, cross)", result)

print("\n=== LOCAL UPDATE IMPLEMENTATION ===")
for name in ("finite_readout221", "rx_step", "rx_update", "rx_core", "rx_coda"):
    fn = globals().get(name)
    if callable(fn) and hasattr(fn, "__code__"):
        c = fn.__code__
        print(name, "arguments", c.co_varnames[:c.co_argcount])
        print("globals", c.co_names)
        print("locals", c.co_varnames)
        print("constants", tuple(x for x in c.co_consts
                                 if isinstance(x, (str, int, float)))[:22])

print("\n=== LOCAL STATE OBJECTS ===")
for name in sorted(list(globals())):
    if name.startswith(("rx_", "r221_", "f221_")):
        v = globals()[name]
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(name, tuple(v.shape))
        elif callable(v):
            c = getattr(v, "__code__", None)
            print(name, "function",
                  c.co_varnames[:c.co_argcount] if c else "")
        elif isinstance(v, dict):
            print(name, "dict", len(v), "keys", list(v)[:5])

print("\n=== FEATURE TOKEN MATCH ===")
for name in ("base238", "nt215", "nt240", "nt"):
    v = globals().get(name)
    if isinstance(v, np.ndarray) and v.shape == (5215, 7, 192):
        print(name, "slot match RMS",
              np.sqrt(np.mean((v[:, :4] - ns258)**2)))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 266\n"
            "Failure diagnosis: P264.T @ (delta_in*B257) has entries up to 61; "
            "the edit reads its own injected directions and creates strong feedback. "
            "Fitted immediate-injection-orthogonal alternatives and inspected local coda/state "
            "implementation to assess transformed feedback. Safe236 remains installed. "
            "Counts: 266 Python runs, 28 forwards.\n")
```