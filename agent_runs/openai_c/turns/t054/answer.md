Numeric-value mixing was factual-safe but did not strengthen the rollouts, so I’ll discard it. I’ll inspect the saved normalization-based designs next: they may let me alter the world summary before code and reading-register terms enter the head. The hypothesis is useful only if a small, centered edit produces stronger dose responses while preserving factual fit.

```python
clear_ops()
replay_ops(repair_candidates["numeric_gain4"])

print("NORMALIZATION/DESIGN VARIABLES")
for k, v in list(globals().items()):
    if isinstance(v, np.ndarray) and any(
        s in k.lower() for s in ["norm", "gamma", "bias", "beta", "diag", "coeff", "sensor", "readout"]
    ):
        print(k, v.shape, "norm", round(float(np.linalg.norm(v)), 6))
print("\nDIAGONAL DESIGNS")
for k, v in diag_designs.items():
    print(k, str(small_description(v))[:1000])
print("\nDOSE MODEL STRUCTURES")
print(str(small_description(dose_models))[:2400])
print("\nSIGNAL STRUCTURES")
print(str(small_description(repair_signal))[:1000])
print("\nRELEVANT TENSORS")
print({k: SHAPES[k] for k in NAMES
       if ("out_norm" in k or "query_norm" in k or "last_" in k
           or k.startswith(("z_head", "core.z_in", "embed_transition.value")))})

print("\nHEAD SENSOR FACTUAL PROJECTIONS")
s_ne = head_ne_design["sensor"]
for event, fit in head_surrogates.items():
    projection = fit["O"] @ s_ne
    print(event, "mean", projection.mean(), "sd", projection.std(),
          "percentiles", np.quantile(projection, [0.01, 0.5, 0.99]))
print("surrogate output responses",
      {e: float(fit["w"] @ head_ne_design["output"])
       for e, fit in head_surrogates.items()})

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 35 / forwards 18–19\n"
            "- Numeric-value mix -2 factual +0.00117, rollouts insulin -2.462/+1.646, NE +0.129/-0.072: rejected for no useful gain.\n"
            "- Restored numeric gain 4. Inspecting saved centered out_norm designs as an alternative to head-input shears.\n"
            "- Python runs: 35; forward passes: 19.\n")
```
