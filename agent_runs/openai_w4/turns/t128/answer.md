The retained work identifies dose-policy variation and provisional lag responses, but it has not validated a sustained-effect repair. I’ll inspect only the candidate weights, transition hooks, and causal summaries now. This should reveal whether the channel can be tested safely without rebuilding the model.

```python
import numpy as np, inspect
print("=== CANDIDATE WEIGHTS ===")
for nm in ["candidate104", "channel_fits112", "channel_fits111",
           "controls112", "controls111", "causal_beta80", "coefficients"]:
    ob = globals().get(nm)
    print(nm)
    if isinstance(ob, dict):
        for kk, vv in ob.items():
            if isinstance(vv, dict):
                print(" ", kk, {
                    k: (tuple(v.shape) if hasattr(v, "shape") else str(v)[:80])
                    for k, v in vv.items()
                })
            elif hasattr(vv, "shape"):
                ar0 = np.asarray(vv)
                print(" ", kk, ar0.shape, "RMS", float(np.sqrt(np.mean(ar0.astype(float)**2))))
            else:
                print(" ", kk, str(vv)[:180])

print("=== FUNCTION DEPENDENCIES ===")
for nm in ["transition73", "capture_transition102", "capture_ep112",
           "capture_q111", "transition_with_b110", "transition_with_z95",
           "gated_innovation94", "rp_linear", "rp_mlp", "probe_means95",
           "head_features_fixed", "build_head_dataset_fixed"]:
    f0 = globals().get(nm)
    if inspect.isfunction(f0):
        print(nm, "file", f0.__code__.co_filename)
        print(" names:", f0.__code__.co_names)
        print(" constants:", repr(f0.__code__.co_consts)[:850])

print("=== RELEVANT OBJECTS ===")
for nm, ob in sorted(list(globals().items())):
    if nm.startswith("_") or inspect.ismodule(ob) or inspect.isfunction(ob):
        continue
    if isinstance(ob, dict) and nm not in {
        "W_repair", "SHAPES", "VOCAB", "__builtins__", "core_trial_cache_repair",
        "core_trial_table_repair", "calibration_candidates80"
    }:
        ks = list(ob.keys())
        if any(s in nm.lower() for s in ["bank", "kernel", "caus", "moment", "state", "gate", "fit", "repair", "sens"]):
            print(nm, "keys", repr(ks[:12])[:180])
    elif hasattr(ob, "columns") and any(s in nm.lower() for s in ["train", "panel", "hist", "target", "tuning"]):
        print(nm, ob.shape, list(ob.columns)[:18])
print("tensor object type:", type(next(iter(W_repair.values()))))
print("Saved edits:", edited())
with open("/work/notes.md", "a") as nf:
    nf.write("\n## Runs 109–115 recovery\n"
             "Run-112 controlled channel decodes q roughly monotonically, but has "
             "factual/non-action leakage 0.18–0.24 RMS. No weights saved. "
             "Runs 113–115 recover retained candidate interfaces; six-lag targets "
             "remain provisional and must not be equated with sustained truth.\n")
```
