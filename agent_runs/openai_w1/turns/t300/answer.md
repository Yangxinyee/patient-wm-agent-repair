The middle-strength projection recovers about 57% of insulin doubling and 73% of norepinephrine doubling on held-out stays, with factual leakage still present. I’ll install a moderate direct-injection candidate and test factual fit first. A loss increase above 0.005 nats would reject it regardless of its dose responses.

```python
import numpy as np
import torch

print("=== RECORD-DERIVED RESPONSE KERNELS ===")
for name in ("he_kernels", "eh_kernels", "kernels251", "targets251"):
    obj = globals().get(name)
    if isinstance(obj, dict):
        print(name)
        for key, value in obj.items():
            print(" ", key, repr(value)[:650])

checkpoint264 = {
    name: arr262(load(name)).copy()
    for name in edited()
}
base_b264 = arr262(load_original("core.b_in.weight")).copy()
P264 = fits263[1.0].copy()
gains264 = np.array([-2.0, 4.0], dtype=np.float64)
delta_b264 = B257 @ np.diag(gains264) @ P264.T
save_edit("core.b_in.weight",
          torch.as_tensor(base_b264 + delta_b264, dtype=torch.float32))

factual_scalar264 = Ef263[hold263] @ P264 * gains264
print("\n=== INSTALLED CANDIDATE ===")
print("Added b_in weight RMS:", np.sqrt(np.mean(delta_b264**2)))
print("Held factual scalar RMS:",
      np.sqrt(np.mean(factual_scalar264**2, axis=(0, 1))))
for j, drug in enumerate(("insulin", "norepinephrine")):
    mask = hold263 & np.asarray(ds_s["has_" + drug], dtype=bool)
    for mult in (0.5, 2.0):
        de = np.asarray(Es259[(drug, mult)], dtype=np.float64) - Ef263
        scalar = de[mask] @ P264 * gains264
        print(drug, mult, "mean local scalar",
              np.mean(scalar, axis=(0, 1)))

print("Factual test stays:", [int(x) for x in ds_eval_ids])
print("Current edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 264\n"
            "Installed reversible direct recurrent injection: original core.b_in plus "
            "B257 @ diag([-2,4]) @ fits263[1].T, retaining safe236 latent/relay weights. "
            "Projection held-out doubling ratios are .568 insulin/.727 NE; factual leakage "
            "requires paired GPU testing. Lower-dose and x3 response recovery is weaker, so "
            "this is an experimental candidate rather than a validated repair. "
            "Counts: 264 Python runs, 27 forwards.\n")
```