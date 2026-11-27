Adding paired evidence improved held-out change prediction substantially, including MAP. These surrogates are now useful enough to test—not yet proof of a repair. I’ll verify the reconstructed state update, then use finite differences to solve for better token directions. An official rollout will still be the decisive test.

```python
import inspect, numpy as np, torch
from pathlib import Path

print("=== LOCAL STATE-UPDATE INTERFACE ===")
for name, limit in [("r59_core", 3000), ("repaired_core_local", 2000),
                    ("lin0", 1000), ("fn93", 1200)]:
    func146 = globals().get(name)
    if func146 is None:
        continue
    print("\n", name)
    try:
        print(inspect.getsource(func146)[:limit])
    except Exception:
        compact_code95(func146, limit=limit)

print("\n=== TRAIN STATE FILE ===")
S146 = dict(np.load("/work/fwd_3/states_original.npz"))
print({k: (v.shape, str(v.dtype)) for k, v in S146.items()})
print("First bundle metadata:",
      list(zip(S146["stay_id"][:8], S146["hour"][:8],
               S146["has_insulin"][:8], S146["has_norepinephrine"][:8])))

print("\n=== LOCAL UPDATE ARRAYS ===")
for name in sorted(list(globals())):
    if name.startswith(("rr33_", "rr34_", "rr35_", "r59_", "r51_")):
        v146 = globals()[name]
        if inspect.isfunction(v146):
            continue
        if hasattr(v146, "shape"):
            print(name, type(v146).__name__, tuple(v146.shape))
        elif isinstance(v146, dict):
            print(name, "keys", list(v146)[:8])

print("\n=== CHANNEL AND CONTRAST SCALE ===")
for name in ["D89", "dr66", "delta88", "coef88", "index66",
             "innov_targets_audit", "causal_positions_audit"]:
    v146 = globals().get(name)
    if v146 is not None and hasattr(v146, "shape"):
        a146 = v146.detach().cpu().numpy() if torch.is_tensor(v146) else np.asarray(v146)
        print(name, a146.shape)
        if a146.ndim == 2 and a146.shape[1] <= 2:
            print("mean/sd:", np.mean(a146, 0), np.std(a146, 0))
print("CF89:")
for c146 in CF89[:3]:
    print({k: (getattr(v, "shape", None) if hasattr(v, "shape") else v)
           for k, v in c146.items()})
print("Current W0 differs from originals:",
      [(n, float(torch.linalg.vector_norm(
          torch.as_tensor(W0[n]) - torch.as_tensor(load_original(n)))))
       for n in candidates90[0.0] if n in W0])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 146: prepare validated-surrogate token routing\n"
            "- Paired held-out delta R2: finger .787, lab .848, arterial .809, cuff .723.\n"
            "- Recovering exact local update and counterfactual feature interfaces for finite-difference linear editing.\n"
            "- No new saved weights; Python runs: 146; forward passes: 10.\n")
```