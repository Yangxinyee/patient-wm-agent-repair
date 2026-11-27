The existing graft uses two gated hidden units, and the synthetic features are available before that gate. I’ll verify the feature conventions and transition checks before replacing its decoder. The critical test is whether the synthetic dose direction survives factual protections; if it does not, a strong repair in this head may be infeasible.

```python
import numpy as np, torch
def shape_summary(obj, depth=0):
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        return ("array", tuple(obj.shape))
    if isinstance(obj, dict):
        return {k: shape_summary(v, depth + 1) for k, v in list(obj.items())[:8]} if depth < 3 else list(obj)[:8]
    if isinstance(obj, (list, tuple)):
        return [shape_summary(v, depth + 1) for v in obj[:5]] if depth < 3 else ("length", len(obj))
    if isinstance(obj, (str, float, int, bool, np.number)):
        return obj
    return type(obj).__name__
for name0 in ["decode_feature_sets", "protected_input_sets", "robust_projectors",
              "lab_simulated", "r139_results", "r140_trials",
              "rank_template", "graft_nodes"]:
    print(name0, shape_summary(globals()[name0]))
print("\nSYNTHETIC METADATA")
print(lab_meta.head(4).to_string(index=False))
print("events", lab_meta.event.value_counts().to_dict() if "event" in lab_meta else list(lab_meta.columns))
print("\nCURRENT GRAFT CONSISTENCY")
for name0 in graft_names:
    now0 = np.asarray(load(name0))
    print(name0, "checkpoint RMSE", float(np.sqrt(np.mean((now0 - graft_checkpoint_beta2[name0])**2))))
print("\nDATA README")
print(open("/data/README.md").read()[:2200])
with open("/work/notes.md", "a") as handle:
    handle.write("\n## Recovery through Python 153; forward 11/50\n"
                 "- Current weights still equal the validated beta=-2 checkpoint.\n"
                 "- Retained 78 synthetic insulin cases and prior-sampled factual head inputs.\n"
                 "- Next: fit synthetic dose contrasts with factual and register/latent protections; "
                 "GPU rollout and factual checks remain required.\n")
```
