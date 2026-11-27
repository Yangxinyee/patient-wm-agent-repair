The record-based rollout targets are provisionally much larger than the model’s: about −14.2/+7.1 for insulin and +4.7/−2.3 for norepinephrine. Before another edit, I’ll check which cached head reconstruction remains valid. Agreement within latent-sampling noise would support using it; large discrepancies would rule it out.

```python
import inspect

print("=== CACHED ORIGINAL RECONSTRUCTION ===")
for ev64 in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    m64 = wr_meta.event.to_numpy() == ev64
    print(ev64, "n", int(m64.sum()),
          "cached mean RMSE",
          float(np.sqrt(np.mean((wr_mean[m64] -
                wr_meta.tool_mean.to_numpy()[m64])**2))))
for k64 in ["wr_base", "wr_mu", "wr_logsd", "wr_fixed", "wr_phi"]:
    a64 = globals()[k64].detach().cpu().numpy()
    print(k64, "quantiles", np.round(np.quantile(a64, [.01, .5, .99]), 4).tolist())

print("\n=== LATENT TENSORS ===")
for n64 in NAMES:
    if "z_prior" in n64 or "z_post" in n64:
        print(n64, SHAPES[n64])

print("\n=== HELPER DEFINITIONS ===")
for k64, lim64 in [("rh_eval", 2400), ("rh_register", 1800), ("wmlp", 800)]:
    fn64 = globals()[k64]
    try:
        text64 = inspect.getsource(fn64)
    except (OSError, TypeError):
        text64 = ("arguments=" + str(fn64.__code__.co_varnames) +
                  "\nconstants=" + str(fn64.__code__.co_consts) +
                  "\nreferenced names=" + str(fn64.__code__.co_names))
    print(k64, text64[:lim64])

print("\n=== INSTRUMENT CACHE ===")
for drug64, cache64 in wr_instruments.items():
    print(drug64, {
        k: (tuple(v.shape) if hasattr(v, "shape") else str(v)[:180])
        for k, v in cache64.items()
    } if isinstance(cache64, dict) else type(cache64).__name__)

print("\nTraining-state stay IDs:",
      wr_meta.stay_id.drop_duplicates().to_numpy()[:8].tolist())
print("Metadata columns:", list(wr_meta))
print("Dictionary names:",
      [k for k, v in list(globals().items())
       if isinstance(v, dict) and not k.startswith("_")])
print("Current edits:", edited())
```