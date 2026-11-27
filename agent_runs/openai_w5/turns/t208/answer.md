Checkpoint alignment did not resolve the replay error. The private coordinates are particularly informative: the GPU states show the intended opposite-valued pairs, but the replay does not. I’ll audit every helper weight, not just the three previously checked. Stale weights would explain this; if all match, the recurrence implementation remains suspect.

```python
import numpy as np, torch

def tensor183(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)

print("=== COMPLETE HELPER WEIGHT AUDIT ===")
stale183 = {}
for label183 in ["W0", "e"]:
    obj183 = globals().get(label183)
    if not isinstance(obj183, dict):
        print(label183, "not a weight dictionary:", type(obj183).__name__)
        continue
    diffs183 = []
    checked183 = 0
    for nm in NAMES:
        if nm not in obj183:
            continue
        a183 = tensor183(obj183[nm])
        b183 = tensor183(load(nm))
        if a183.shape != b183.shape:
            diffs183.append((nm, "shape mismatch"))
        else:
            checked183 += 1
            d183 = float(torch.max(torch.abs(a183-b183)))
            if d183 > 0:
                diffs183.append((nm, d183))
    stale183[label183] = diffs183
    print(label183, "checked", checked183, "different", len(diffs183))
    for item183 in diffs183[:35]:
        print(item183)

print("\n=== PRIVATE ROW INVARIANTS ===")
pairs183 = [(61,21), (15,66)]
for nm in ["core.b_in.weight", "core.c_out.weight",
           "core.prelude.0.attn.out.weight",
           "core.blocks.0.0.attn.out.weight", "core.blocks.0.0.cross.out.weight",
           "core.blocks.0.0.ffn.2.weight", "core.blocks.0.1.attn.out.weight",
           "core.blocks.0.1.cross.out.weight", "core.blocks.0.1.ffn.2.weight"]:
    a183 = tensor183(load(nm))
    print(nm, "row norms", [float(a183[k].norm()) for k in [61,21,15,66]],
          "pair-sum norms", [float((a183[a]+a183[b]).norm()) for a,b in pairs183])

print("\n=== LINEAR/MLP HELPER REFERENCES ===")
for nm in ["wm_lin", "wm_mlp", "lin0", "mlp0", "norm0"]:
    fn183 = globals().get(nm)
    if callable(fn183) and hasattr(fn183, "__code__"):
        print(nm, "args", fn183.__code__.co_varnames[:fn183.__code__.co_argcount],
              "names", fn183.__code__.co_names,
              "constants", [v for v in fn183.__code__.co_consts
                            if isinstance(v, (str, int, float))])

if stale183.get("W0"):
    for nm in NAMES:
        W0[nm] = tensor183(load(nm)).clone()
    print("\nReloaded all W0 weights from current edits.")
    for j183 in [3,4,30,31]:
        p183 = wm_step_boundary176(
            j183, include_current=best182[2],
            age_mode="input", age_units="seconds"
        ).detach().cpu().numpy().reshape(4,192)
        t183 = s4["x"][j183,192:960].reshape(4,192)
        print(j183, "ordinary RMSE",
              float(np.sqrt(np.mean((p183[:,ordinary182]-t183[:,ordinary182])**2))),
              "CPU private", p183.mean(0)[idx182].tolist(),
              "GPU private", t183.mean(0)[idx182].tolist())
```