The direct-token directions avoid changing attention weights, and their local factual distortion is much smaller than the rejected routing edits. Their dose responses are still weak, so I’ll provisionally install gain 0.3 and test it. It is useful only if factual NLL stays within the limit and rollout effects improve.

```python
import pandas as pd
candidate257 = {k: np.array(v, dtype=np.float32, copy=True)
                for k, v in np.load("/work/neutral209.npz").items()}
gain257 = 0.3
for drug257, column257 in [("insulin", 16), ("norepinephrine", 12)]:
    vector257 = np.asarray(mean_directions216[drug257], dtype=np.float32)
    candidate257["z_in.weight"][:, column257] += gain257 * vector257
for name257 in list(edited()):
    drop_edit(name257)
for name257, tensor257 in candidate257.items():
    save_edit(name257, tensor257)
np.savez("/work/token257.npz", **candidate257)

fact257 = pd.read_csv("/work/fwd_18/factual_original.csv")
stay_col257 = next(c for c in fact257.columns if "stay" in c.lower())
dev257 = sorted(fact257[stay_col257].unique().tolist())
print("INSTALLED gain", gain257, "edits", edited())
print("DEVELOPMENT STAYS", dev257)
print("FORWARD18 SUMMARY", Path("/work/fwd_18/summary.json").read_text()[:1700])
print("PRIVATE COLUMN NORMS",
      {c: round(float(np.linalg.norm(candidate257["z_in.weight"][:, c])), 5)
       for c in [16, 12]})
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 257 — provisional low-gain direct-token response\n"
            "- Installed token257: neutral209 plus mean_directions216 at gain 0.3 in z_in columns 16 and 12.\n"
            "- No attention weights changed. Local factual KL proxy was 0.000685, but this is not certification.\n"
            "- Requires matched GPU factual dNLL <= .005 and beneficial dose rollouts before acceptance.\n")
```