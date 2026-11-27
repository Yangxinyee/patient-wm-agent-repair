The estimated responses persist for several hours, so repairing only the immediate prediction would be insufficient. Before using the retained counterfactual features, I’ll run an essential alignment check: a dose multiplier of one must reproduce the factual computation. A mismatch would invalidate the earlier head-fit conclusions and need fixing first.

```python
import numpy as np, torch, dis
from pathlib import Path

torch.set_num_threads(8)
with torch.no_grad():
    factual_logits101 = (
        mlp63(input_fact96, "attr_heads.value_bin")
        + torch.as_tensor(trans_train67)[None, :, :]
    )
    factual_probs101 = factual_logits101.softmax(-1)
    factual_means101 = (
        factual_probs101 * torch.as_tensor(centres_train67)[None, :, :]
    ).sum(-1)

    baseline_checks101 = {}
    for drug101 in ("insulin", "norepinephrine"):
        c101 = make_cf81(drug101, 1.0, batch_size=96)
        rows101 = np.asarray(c101["rows"], dtype=int)
        logits101 = torch.as_tensor(c101["logits"])
        means101 = torch.as_tensor(c101["means"])
        dl101 = logits101 - factual_logits101[:, rows101]
        dm101 = means101 - factual_means101[:, rows101]
        baseline_checks101[drug101] = {
            "readings": len(rows101),
            "logit_RMS": float(dl101.square().mean().sqrt()),
            "mean_RMS": float(dm101.square().mean().sqrt()),
            "mean_max_abs": float(dm101.abs().max()),
        }
    print("Factor-one equivalence:", baseline_checks101)

print("\n=== MASK HANDLING IN COUNTERFACTUAL FUNCTION ===")
ins101 = list(dis.get_instructions(make_cf81))
shown101 = set()
for j101, op101 in enumerate(ins101):
    if op101.argval in ("padding", "transition64"):
        for k101 in range(max(0, j101 - 3), min(len(ins101), j101 + 5)):
            if k101 not in shown101:
                p101 = ins101[k101]
                print(p101.offset, p101.opname, p101.argrepr)
                shown101.add(k101)

print("\n=== POSTERIOR AND LATENT SCALE ===")
for name101 in ("z_in.weight", "z_head.weight", "z_post.2.weight"):
    a101 = np.asarray(load_original(name101))
    print(name101, "RMS", float(np.sqrt(np.mean(a101*a101))),
          "column norms", np.round(np.linalg.norm(a101, axis=0)[:32], 3))

with Path("/work/notes.md").open("a") as f101:
    f101.write(
        "\n## Run 101 / counterfactual factor-one falsification\n"
        f"Factor-one checks: {baseline_checks101}. "
        "Require factual equivalence before interpreting retained CF sensitivities or fitting edits. "
        "No weights edited.\n"
    )
```
