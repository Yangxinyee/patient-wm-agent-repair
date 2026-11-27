The affine readout candidates are still available; the inspection failed only when summarizing a string array. I’ll stage the candidate that previously passed local validation, then audit it on GPU. If its factual NLL rises by more than 0.005, I’ll remove it rather than rely on the local estimate.

```python
import numpy as np, torch
from pathlib import Path

assert X179.shape[-1] == 193 and np.allclose(X179[..., -1], 1.0)
print("LOCAL FACTUAL RECHECK")
for lam185 in [0.1, 1.0]:
    corr185 = readout179[lam185]
    logits185 = L179 + np.einsum("bnd,dk->bnk", X179, corr185)
    dnll185 = nll179(logits185) - original_nll179
    print(lam185,
          "train delta", float(np.mean(dnll185[fitmask179])),
          "held-out delta", float(np.mean(dnll185[valmask179])),
          "all delta", float(np.mean(dnll185)))

candidate185 = readout179[1.0].copy()
head_names185 = ["attr_heads.value_bin.2.weight", "attr_heads.value_bin.2.bias"]
head_original185 = {n: np.asarray(load_original(n)).copy() for n in head_names185}
assert head_original185[head_names185[0]].shape == candidate185[:-1].T.shape
head_weight185 = head_original185[head_names185[0]] + candidate185[:-1].T
head_bias185 = head_original185[head_names185[1]] + candidate185[-1]
for name185, value185 in zip(head_names185, [head_weight185, head_bias185]):
    save_edit(name185, value185.astype(np.float32))
    w63_cache[name185] = torch.as_tensor(value185, dtype=torch.float32)

print("\nSTAGED EDITS", edited())
print("FACTUAL AUDIT STAYS", list(audit_stays169))

print("\nCORE TEST INTERFACES")
for k185 in ["make_cf81", "step63", "head_inputs96", "prediction179"]:
    if k185 in globals():
        f185 = globals()[k185]
        print(k185, f185.__code__.co_varnames, f185.__code__.co_names)

with Path("/work/notes.md").open("a") as f185:
    f185.write(
        "\n## Run 185 / conservative value-readout staged\n"
        "Recovered original rollout: I x2 -3.680, I half +2.018; NE x2 +0.465, "
        "NE half -0.871. Accepted records-based hour-12 targets remain approximately "
        "-12.77,+6.38,+4.50,-2.25; rejected early SSM targets are not used. "
        "Staged readout179[1.0] in value_bin.2 weight and bias on top of the "
        "three quiet-wire edits, pending paired factual GPU validation. "
        "No upstream controller or other head was changed.\n"
    )
```
