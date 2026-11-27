The probe has a promising near-exact fit: its scalar error is about \(1.5\times10^{-6}\). That points to an incorrect register layout or age scaling in the earlier head reconstructions. I’ll resolve the remaining presence/status fields next. Matching the probe’s known coefficients would confirm the layout; a mismatch would leave it unverified.

```python
import itertools
import numpy as np
import torch

print("=== BEST REGISTER-PROBE FITS ===")
for result in r90_results[:6]:
    print(result)

p95 = np.asarray(r89_pw, dtype=np.float64)
s95 = np.asarray(load_original("last_status.weight"), dtype=np.float64)
n95 = np.asarray(load_original("last_numeric.weight"), dtype=np.float64)
print("\nprobe numeric coefficients:", n95.T @ p95)
print("probe status projections:", s95 @ p95)

best95 = r90_results[0]
coef95 = np.asarray(best95["coefficients_has_age_z"])
if best95["status"] == "has_last":
    has95 = coef95[0] + (s95[1] - s95[0]) @ p95
else:
    has95 = coef95[0]
print("presence coefficient assuming measurement status stays zero:", has95)

flag_indices95 = [0, 1, 2, 4, 5, 7]
matches95 = []
for bits95 in itertools.product([0, 1], repeat=len(flag_indices95)):
    value95 = sum(bits95[j] * (n95.T @ p95)[i]
                  for j, i in enumerate(flag_indices95))
    matches95.append((abs(value95 - has95),
                      [i for i, bit in zip(flag_indices95, bits95) if bit],
                      value95))
print("closest binary-presence layouts:", sorted(matches95)[:5])

print("\n=== AGE AND CACHED INPUT DETAILS ===")
print(r90_meta[["hour", "event", "dt", "has_last", "age", "last_z"]]
      .head(10).to_string(index=False))
for key95, arr95 in r90_age_options.items():
    print(key95, np.asarray(arr95)[:10])
print("last tested age feature:", np.asarray(r90_af_used)[:10])

print("\n=== DETERMINISTIC RECONSTRUCTION CACHE ===")
for name95 in ["r91_W", "r91_fixed", "r91_lb", "r91_tool", "r91_tm",
               "r91_tn", "wr_h", "wr_out", "wr_dt", "wr_meta",
               "rh_out", "rh_h", "rh_dt", "probe83_factual",
               "probe84_df", "states85"]:
    obj95 = globals().get(name95)
    if isinstance(obj95, dict):
        print(name95, "dict", len(obj95), repr(list(obj95)[:8]))
    elif hasattr(obj95, "shape"):
        print(name95, type(obj95).__name__, tuple(obj95.shape))
    else:
        print(name95, type(obj95).__name__, repr(obj95)[:100])

print("r91_mlp defaults:", r91_mlp.__defaults__)
print("r91_norm names:", r91_norm.__code__.co_names)
print("CURRENT EDITS:", edited())
```
