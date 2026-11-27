The routing test did not fix the reading mismatch; the current atom flags already treat END atoms as observations. I’ll now compare bundle counts and posterior errors by bundle type. Agreement on drug-only bundles would support testing a limited drug-posterior edit, even if reading reconstruction remains imperfect.

```python
import numpy as np
import torch
from pathlib import Path

print("=== ATOM HELPER ===")
print("defaults:", r36_atom.__defaults__)
print("names:", r36_atom.__code__.co_names)
print("constants:", [str(x)[:160] for x in r36_atom.__code__.co_consts])
print("numeric_note:", VOCAB["numeric_note"])
print("numeric examples:", str(VOCAB["numeric"][:2])[:1300] if isinstance(VOCAB["numeric"], list)
      else str(list(VOCAB["numeric"].items())[:2])[:1300])

r80_parts = []
for _path in ("/work/fwd_2/states_original.npz", "/work/fwd_4/states_original.npz"):
    with np.load(_path) as _f:
        print(_path, [(k, _f[k].shape) for k in _f.files])
        r80_parts.append({k: _f[k] for k in _f.files})
r80_raw = {k: np.concatenate([p[k] for p in r80_parts]) for k in r80_parts[0]}
r80_lookup = {z.tobytes(): i for i, z in enumerate(r80_raw["x"][:, -32:])}
r80_aligned = np.array([r80_lookup[z.numpy().tobytes()] for z in r79_target])
r80_counts = np.bincount(r79_owner.numpy(), minlength=r79_n)
print("=== BUNDLE COUNT AGREEMENT ===")
print("exact fraction:", np.mean(r80_counts == r80_raw["n_atoms"][r80_aligned]))
print("differences/counts:", np.unique(r80_counts - r80_raw["n_atoms"][r80_aligned],
                                    return_counts=True))
print("alignment maximum posterior error:",
      np.max(np.abs(r80_raw["x"][r80_aligned, -32:] - r79_target.numpy())))

r80_e, r80_ec = r79_encode(~r79_action, "encode", "gelu")
r80_u, r80_uc = r79_encode(r79_action, "encode_action", "gelu")
r80_input = torch.cat([r74_contexts["core.out_norm/gap/gelu"], r80_e, r80_u], 1)
r80_z = r79_mlp(r80_input, "z_post", "gelu")[:, :32]
r80_err = ((r80_z - r79_target) ** 2).mean(1).sqrt().numpy()
r80_end_counts = np.bincount(r79_owner[r79_end].numpy(), minlength=r79_n)
r80_hour = r80_raw["hour"][r80_aligned]
for _label, _mask in (
    ("drug-only", r79_drug_rows.numpy() & ~r79_read_rows.numpy()),
    ("reading, no END", r79_read_rows.numpy() & (r80_end_counts == 0)),
    ("reading, with END", r79_read_rows.numpy() & (r80_end_counts > 0)),
    ("END-only", ~r79_read_rows.numpy() & ~r79_drug_rows.numpy() & (r80_end_counts > 0)),
):
    print(_label, "n", int(_mask.sum()), "error quantiles",
          np.round(np.quantile(r80_err[_mask], [0, .5, .9, 1]), 6) if _mask.any() else [])
print("=== EXPECTED-DOSE FIT LAYOUT ===")
for _drug, _fit in rr51_assignment.items():
    print(_drug, {str(k): (tuple(v.shape) if isinstance(v, (np.ndarray, torch.Tensor))
                          else str(v)[:180]) for k, v in _fit.items()})
print("=== STATE/EDIT HELPERS ===")
print([n for n in sorted(list(globals()))
       if any(s in n.lower() for s in ("neuron", "importance", "kernel", "impulse", "target_shift"))])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 80\nEND routing was not the source of reading mismatch. Checking exact exported bundle counts and drug-only posterior agreement before limiting an edit to action innovations. No edits. Counts: 80 Python runs, 6 forward passes.\n")
```