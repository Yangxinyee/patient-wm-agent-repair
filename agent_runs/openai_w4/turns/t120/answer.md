The records support an immediate insulin effect near −2 glucose units per rate unit and a norepinephrine effect near +3.17 MAP units, subject to the stated identification assumptions. These are not yet the sustained rollout targets. I’ll now recover the candidate settings and transition interface so I can test factual disturbance directly rather than judge it from hidden-state error alone.

```python
import inspect

print("--- candidate changes ---")
for n, t in candidate104.items():
    old = torch.as_tensor(load_original(n), dtype=torch.float32)
    new = torch.as_tensor(t, dtype=torch.float32)
    print(n, tuple(new.shape), "delta RMS", round(float((new-old).square().mean().sqrt()), 6))

print("\n--- candidate settings ---")
for n in ("head104", "keep104", "latent_axis104", "carrier104", "signature104",
          "lambda104", "gamma104", "bias104", "pencomp104", "vec104"):
    obj = globals().get(n)
    if hasattr(obj, "shape"):
        ar = np.asarray(obj)
        print(n, ar.shape, "norm", float(np.linalg.norm(ar)), "first", ar.ravel()[:5])
    else:
        print(n, obj)

print("\n--- transition and data interfaces ---")
for n in ("transition73", "transition_with_z95", "capture_transition102",
          "probe_means95", "pre_context_repair"):
    fn = globals().get(n)
    if callable(fn):
        print(n, inspect.signature(fn))
        print(" dependencies:", fn.__code__.co_names)
print("repair_bank keys:", list(repair_bank))
for n in ("probe_indices95", "probe_data95", "probe_fit95", "head_table103",
          "head_checks103", "kernel55", "provisional_kernels", "effects89",
          "raw104", "S104", "NN104", "P104", "target_attention103"):
    obj = globals().get(n)
    if isinstance(obj, pd.DataFrame):
        print(n, obj.shape, list(obj.columns))
        if len(obj) < 16:
            print(obj.to_string(index=False))
    elif isinstance(obj, dict):
        print(n, "dict keys", list(obj)[:16])
    elif hasattr(obj, "shape"):
        print(n, type(obj).__name__, tuple(obj.shape))
    else:
        print(n, repr(obj)[:220])
```
