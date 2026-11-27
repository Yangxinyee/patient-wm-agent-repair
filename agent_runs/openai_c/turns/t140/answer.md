The recurrent gain failed factual fit: NLL increased by 0.02422 nats. I’ll remove it and try a drug-specific alternative—tripling contrasts between nonzero dose/rate bin embeddings while preserving the absent-bin row and their common component. Stronger rollouts would support this approach; saturation or wrong-sign effects would reject it.

```python
from scipy.linalg import helmert

undo()
assert len(ops()) == 1, ops()

bin_contrast_gain = 3.0
for name in ["embed.bins.dose.weight", "embed.bins.rate.weight"]:
    n_rows, width = SHAPES[name]
    print(name, SHAPES[name])
    basis = helmert(n_rows - 1, full=False).T
    for j in range(basis.shape[1]):
        v_bin = np.zeros(n_rows)
        v_bin[1:] = basis[:, j]
        rank1(name, v_bin, bin_contrast_gain - 1.0, side="out")

bin_gain3_ops = ops()
print("INSTALLED", len(bin_gain3_ops), "operations")
print("Nonzero-bin contrasts x3; absent rows and common components unchanged.")

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 99 / forwards 37–38: recurrent gain rejected; bin pilot\n"
            "- Recurrent input x2: insulin -3.768/+2.902; NE +0.259/-0.131.\n"
            "- Factual NLL increase +0.02422: unacceptable. Removed recurrent scaling.\n"
            "- Numeric gain 4 retained; dose/rate nonzero-bin contrasts amplified x3.\n"
            "- Preserves absent bins and common nonzero-bin embeddings; measurement bins unchanged.\n"
            "- Factual safety and rollout behavior unknown. Forward passes used: 38; 12 remain.\n")
```
