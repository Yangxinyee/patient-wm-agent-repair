The later training data contain paired original and insulin-repaired states. I’ll extract nonlinear features on the repaired states and verify their factual reconstruction before fitting. Exact reconstruction and held-out sensitivity to both NE dose changes would support a trial; poor alignment or persistent asymmetry would refute it.

```python
from pathlib import Path
import numpy as np, torch, inspect

hidden_base242, raw_base242 = prelude_hidden93(
    h_base191, obs191, act191, z_base191
)
E_check242 = core_norm(
    torch.from_numpy(raw_base242), "core.prelude_norm"
).numpy()
E_reference242 = (
    E_base191.detach().cpu().numpy()
    if isinstance(E_base191, torch.Tensor) else np.asarray(E_base191)
)
print("FEATURE RECONSTRUCTION",
      "hidden", hidden_base242.shape,
      "E_RMS", float(np.sqrt(np.mean((E_check242-E_reference242)**2))),
      "E_max", float(np.max(np.abs(E_check242-E_reference242))))

print("\nPOSTERIOR RECONSTRUCTION ERRORS")
for label, errors in (("original", zerr_orig191), ("baseline", zerr_base191)):
    a = errors.detach().cpu().numpy() if isinstance(errors, torch.Tensor) else np.asarray(errors)
    for subset, mask in (("all", np.ones(len(a), dtype=bool)),
                         ("drug", np.asarray(isdrug191, dtype=bool))):
        print(label, subset, "RMS", float(np.sqrt(np.mean(a[mask]**2))),
              "max", float(np.max(np.abs(a[mask]))))

print("\nTRAINING CONTRAST CONTRACTS")
for key, data in cf_train192.items():
    print(key)
    for field, value in data.items():
        print(" ", field, tuple(value.shape) if hasattr(value, "shape")
              else type(value).__name__)
    for field in ("rate", "dose_change"):
        if field in data:
            a = np.asarray(data[field])
            print(" ", field, "quantiles", np.quantile(a, [0, .25, .5, .75, 1]))

print("\nENTRY REPRESENTATIONS")
drug_indices242 = np.flatnonzero(np.asarray(isdrug191, dtype=bool))
for i in [int(first_indices198[0]), int(drug_indices242[0]), int(drug_indices242[-1])]:
    key = (int(sid191[i]), int(minute191[i]))
    entry = entries191[key]
    print("bundle", i, key, "count", counts191[i],
          "entry", repr(entry)[:900])

print("\nENCODING HELPER CONTRACTS")
for name in ("inputs191", "posterior94", "recon_record_atom",
             "single_encode_batch", "recon_sets"):
    fn = globals().get(name)
    if callable(fn) and hasattr(fn, "__code__"):
        c = fn.__code__
        print(name, inspect.signature(fn),
              "locals", c.co_varnames,
              "names", c.co_names,
              "constants", [x for x in c.co_consts
                            if isinstance(x, (str, int, float))])

print("\nFOLD COUNTS", np.unique(foldF197, return_counts=True))
print("initial admission bundles", int(np.sum(initial_admission198)))
with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 242: training nonlinear features and alignment audit\n"
        "Extracted prelude GELU features on baseline77 training states. "
        "Checking normalized reconstruction, posterior errors and synthetic "
        "dose-contrast encoding contracts before closed-form fitting. "
        "Joint236 unchanged.\n"
        "Python runs: 242; forwards: 29.\n"
    )
```