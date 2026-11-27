Doubling the correction has an estimated NLL cost of 0.00262 overall, but 0.00645 on the smaller held-out subset—a warning that it may not generalize. I’ll install it provisionally and check GPU factual fit. If the measured cost exceeds 0.005, I’ll restore the saved candidate or reduce the strength.

```python
import torch

r112_strength = 2.0
r112_candidate = {n: t.clone() for n, t in r107_backup.items()}
r112_candidate["attr_heads.value_bin.2.weight"][:, pair104] = (
    r112_strength
    * r107_backup["attr_heads.value_bin.2.weight"][:, pair104]
)

for name112, tensor112 in r112_candidate.items():
    save_edit(name112, tensor112)

with open("/work/notes.md", "a") as f:
    f.write("""
## Run 112: provisional insulin strength 2
Scaled only the two correction output columns; gate, decoder, and
removed-neuron mean compensation are unchanged.
Local delta NLL at strength 2: all +0.002618, training +0.001343,
held-out +0.006454. Held-out result is a generalization warning.
This is a trial, not an accepted replacement. Saved strength-1 candidate
remains available for immediate restoration. GPU factual check is next.
""")

print("Installed insulin correction strength:", r112_strength)
print("Edited tensors:", edited())
print("Bias unchanged:",
      torch.equal(torch.as_tensor(load("attr_heads.value_bin.2.bias")),
                  r107_backup["attr_heads.value_bin.2.bias"]))
```