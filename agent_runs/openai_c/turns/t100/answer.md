The earlier selective edits relied on observational readout approximations and produced little rollout improvement. I’ll distinguish those from measured responses to weight changes and inspect the untouched pathways. A useful direction should respond to drug information while varying little on factual records; otherwise amplifying it is likely to damage fit.

```python
print("=== TARGET TENSOR SHAPES ===")
for name in NAMES:
    if any(s in name for s in [
        "attr_heads.value", "query_norm", "last_numeric", "embed_transition.value",
        "encode_action", "core.coda", "core.c_out", "z_head", "z_prior",
        "core.prelude.0.self_attn"
    ]):
        print(name, SHAPES[name])

print("\n=== SAVED CANDIDATE KEYS ===")
print(list(repair_candidates))
print("\n=== HEAD SURROGATE STRUCTURE ===")
for event, fit in head_surrogates.items():
    print(event, {k: (v.shape if isinstance(v, np.ndarray) else
                       list(v)[:6] if isinstance(v, dict) else v)
                  for k, v in fit.items()
                  if k not in {"model", "models", "estimator"}})

print("\n=== LATENT / HEAD SENSOR DIAGNOSTICS ===")
for key in ["latent_candidate_basis", "latent_sensor_matrix", "latent_sv",
            "head_ne_design", "direction_diagnostics", "repair_signal",
            "repair_white"]:
    value = globals().get(key)
    if isinstance(value, np.ndarray):
        print(key, value.shape, "norm =", float(np.linalg.norm(value)))
    elif isinstance(value, dict):
        print(key, repr(value)[:1200])

print("\n=== REMAINING STATE / CONTROLLED-RESPONSE OBJECTS ===")
for k, v in list(globals().items()):
    if k.startswith("_"):
        continue
    if any(s in k.lower() for s in ["59", "60", "controlled", "head_response",
                                   "paired", "response_fit", "state24", "tune_states"]):
        print(k, v.shape if isinstance(v, (np.ndarray, pd.DataFrame)) else
              list(v)[:8] if isinstance(v, dict) else type(v).__name__)
```
