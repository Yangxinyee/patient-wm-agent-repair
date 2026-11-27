The posterior amplification barely changed any rollout response. That rejects this pathway as a useful repair at the tested strength. I’ll restore it and inspect the direct action-to-state pathway instead; the next candidate needs measurable dose sensitivity, not merely a larger latent perturbation.

```python
from pathlib import Path
import numpy as np

drop_edit("z_post.2.weight")
drop_edit("z_post.2.bias")
print("ACTIVE EDITS:", edited())
print("\nDATA README:\n", Path("/data/README.md").read_text()[:3200])
print("\nCONFIG:", VOCAB.get("config"))
print("\nCORE AND EMBEDDING TENSORS:")
for n in NAMES:
    if n.startswith(("core.", "encode.", "encode_action.", "embed.categorical.")):
        print(n, SHAPES[n])

print("\nRECONSTRUCTED ATOM COLUMNS:", repair_atoms.columns.tolist())
print(repair_atoms.head(10).to_string(index=False))
print("\nONE RAW STAY:")
print(train[train.stay_id == train.stay_id.iloc[0]].head(18).to_string(index=False))
print("\nPOSTERIOR RECONSTRUCTION ERROR BY BUNDLE TYPE:")
z_err = np.sqrt(np.mean(
    (np.asarray(post_z0)[:, :32] - np.asarray(post_truth)) ** 2, axis=1))
for label, mask0 in [
    ("action", repair_has_action & post_valid),
    ("end", repair_has_end & ~repair_has_action & post_valid),
    ("other", ~repair_has_end & ~repair_has_action & post_valid)
]:
    print(label, int(mask0.sum()), np.quantile(z_err[mask0], [.5, .9, 1]))

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Forward 10–11; run 65: rejected posterior probe\n"
            "- Factual NLL change -0.00003, but rollout changes negligible: "
            "insulin -3.665/+2.041; NE +0.472/-0.837.\n"
            "- Restored both z_post tensors. Posterior amplification is not "
            "an effective repair at this strength.\n"
            "- Next inspect direct action state transmission and dose-contrast "
            "features at the value readout.\n"
            "- Python runs: 65. Forward passes: 11. No active edits.\n")
```
