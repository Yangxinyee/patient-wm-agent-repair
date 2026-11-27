Earlier head shears also acted on the last-reading fields. I’ll check an alternative: transform those fields inversely and compensate the input encoders, leaving their contributions unchanged while editing only the head’s query contribution. This algebraic separation needs verification; lower factual cost and stronger dose responses would support it.

```python
import numpy as np
from pathlib import Path

I105 = np.eye(192)
print("HEAD TRANSFORM ALGEBRA")
for name in ("S72", "target73", "factor73", "implemented73", "head_actual"):
    a = np.asarray(globals()[name])
    print(name, "distance from identity:", np.linalg.norm(a - I105),
          "rank of change:", np.linalg.matrix_rank(a - I105, tol=1e-7))
print("S72 vs T72 R72^T:", np.linalg.norm(S72 - T72 @ R72.T))
print("S72 vs T72 W72:", np.linalg.norm(S72 - T72 @ W72))
print("S72 squared:", np.linalg.norm(S72 @ S72))
print("target73 vs I + .2*S72:",
      np.linalg.norm(target73 - (I105 + 0.2 * S72)))
print("head basis orthogonality:",
      np.linalg.norm(head_basis73.T @ head_basis73 - np.eye(head_basis73.shape[1])))

print("\nEMBEDDING / REGISTER / FIRST-ENCODER TENSORS")
for name in NAMES:
    if (name.startswith("embed.") or name.startswith("last_status")
        or name.startswith("last_numeric")
        or name.startswith("encode.mlp1.0")
        or name.startswith("encode_action.mlp1.0")):
        print(name, SHAPES[name])

print("\nTRANSFORM HELPER DETAILS")
for fn in (apply_lowrank_left_transform, apply_lowrank_right_transform,
           effective_transform):
    print(fn.__name__, fn.__code__.co_names, fn.__code__.co_consts)

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Python run 105: register-compensated head design\n"
            "- Numeric gain 4 unchanged; no new operations installed.\n"
            "- Proposed inverse basis change of atom embeddings and last-reading registers,\n"
            "  with compensating first-layer encoder transforms, to isolate a query-only head shear.\n"
            "- Auditing tensor coverage and transform algebra before installation.\n"
            "- Forward passes used: 40; 10 remain.\n")
```