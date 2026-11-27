The recurrence is still approximate, and the diagnostic arrays contain missing values. The apparent finite-dose failure could therefore reflect the readout calculation rather than the direction itself. I’ll inspect that wrapper and compare responses only on valid, matched bundles. Stable, correctly indexed responses would support a small GPU test; persistent disagreement would reject this approach.

```python
import dis
import numpy as np
import torch

def compact_code73(func):
    lines = []
    current_line = None
    parts = []
    for ins in dis.get_instructions(func):
        if ins.starts_line is not None and ins.starts_line != current_line:
            if parts:
                lines.append(f"{current_line}: " + " | ".join(parts))
            current_line = ins.starts_line
            parts = []
        if ins.opname not in ("RESUME", "CACHE", "NOP"):
            parts.append(ins.opname + (f"({ins.argrepr})" if ins.argrepr else ""))
    if parts:
        lines.append(f"{current_line}: " + " | ".join(parts))
    return "\n".join(lines)

print("=== READOUT IMPLEMENTATION ===")
print(compact_code73(next_means61))
print("\n=== RESPONSE WRAPPER ===")
print(compact_code73(response_pair68))

print("\n=== VALID-MATCHED RESPONSE DIAGNOSTICS ===")
valid_jac73 = np.isfinite(jac68).all(axis=(1, 2))
print("valid Jacobians:", valid_jac73.sum(), "/", len(valid_jac73))
print("linear response mean:",
      np.einsum("nrd,dk->nrk", jac68[valid_jac73], V68).mean(axis=0))
valid_finite73 = np.isfinite(finite_response68).all(axis=(1, 2))
print("valid finite responses:", valid_finite73.sum(), "/", len(valid_finite73))
print("finite response mean:", finite_response68[valid_finite73].mean(axis=0))
print("first valid sample rows:", sample68[valid_jac73][:5])
print("reconstruction globals:", wm_kv_layout, wm_memory_mode)
print("W0 z-post/head shapes:",
      {k: tuple(v.shape) for k, v in W0.items()
       if k.startswith(("z_post", "z_in", "z_head"))})

print("\n=== ONE READOUT'S ACTUAL RETURN SHAPES ===")
if valid_jac73.any():
    row73 = int(sample68[np.flatnonzero(valid_jac73)[0]])
    hsaved73 = torch.from_numpy(s4["x"][row73, 192:960]).reshape(1, 4, 192)
    ret73 = next_means61(row73, hsaved73)
    for i73, val73 in enumerate(ret73):
        print(i73, type(val73).__name__, getattr(val73, "shape", None),
              repr(val73)[:500])
print("Current edits:", edited())
```