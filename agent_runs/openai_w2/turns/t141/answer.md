The records support much stronger dose effects than the original rollout, although their accumulation remains uncertain. I’ll start with a conservative insulin-only correction, using a direction that changes glucose while cancelling its average MAP response. Local perturbations will check the sign and scale before a factual-fit and rollout test.

```python
print("=== DOSE TARGET UNITS ===")
for name125 in ("dose_actual_repair", "posterior_cf_targets_repair",
                "fit_mask", "check_mask", "target_scale", "record_targets",
                "aligned_targets", "original_dose_table"):
    obj125 = globals().get(name125)
    print(name125, describe115(obj125))
    if isinstance(obj125, np.ndarray):
        print(" sample:", obj125[:3], "range:",
              np.nanmin(obj125), np.nanmax(obj125))
    elif isinstance(obj125, dict):
        for key125, val125 in list(obj125.items())[:4]:
            print(" ", key125, describe115(val125))
            if isinstance(val125, np.ndarray):
                print("  sample", val125[:2])

print("\n=== STATE INDEX IDENTIFIERS ===")
print(", ".join(k for k, v in list(globals().items())
      if not k.startswith("_") and not callable(v)
      and any(s in k.lower() for s in ("indices", "idx", "rows", "stays")))[-2300:])

print("\n=== B_IN TENSOR SHAPES ===")
print([(n, SHAPES[n]) for n in NAMES if n.startswith("core.b_in")])

# Recompute derivatives explicitly with respect to the output of b_in.
eps125 = 0.25
J125 = np.empty_like(J120)
for d125 in range(192):
    perturb125 = torch.zeros_like(hpre119)
    perturb125[:, :, d125] = eps125 * delta119[d125]
    plus125 = read_hpre120(hpre119 + perturb125)["mean"]
    minus125 = read_hpre120(hpre119 - perturb125)["mean"]
    J125[:, d125] = (plus125 - minus125) / (2 * eps125)

events125 = probe_head110["event"].to_numpy()
A125 = np.stack([J125[events125 == e].mean(axis=0)
                for e in ("glucose_finger", "map_arterial")])
V125 = A125.T @ np.linalg.inv(A125 @ A125.T)
print("\n=== STEERING CHECK ===")
print("Derivative relative difference from J120:",
      np.linalg.norm(J125-J120)/np.linalg.norm(J125))
print("direction norms:", np.linalg.norm(V125, axis=0))
print("linear cross-response:", A125 @ V125)
for j125 in range(2):
    pert125 = (torch.tensor(V125[:, j125], dtype=torch.float32)
               * delta119)[None, None, :]
    out125 = read_hpre120(hpre119 + pert125)["mean"]
    base125 = read_hpre120(hpre119)["mean"]
    print("direction", j125,
          {e: float((out125-base125)[events125 == e].mean())
           for e in ("glucose_finger", "map_arterial")})

fit125 = prelude_fits112["E"][0.3]
Wdose125 = fit125["B"][1:] / fit125["sd"][:, None]
bdose125 = fit125["B"][0] - fit125["mu"] @ Wdose125
print("chosen raw coefficient norms:", np.linalg.norm(Wdose125, axis=0))
print("chosen intercept:", bdose125)
print("current edits:", edited())
```
