All four factual batches pass; their pooled deterioration is approximately 0.0032 nats. I’ll preserve this validated fallback, slightly reduce insulin strength toward its record-based target, and increase the MAP correction by 60%. This trade-off is worthwhile only if official checks show better MAP responses while retaining factual eligibility.

```python
from pathlib import Path
import numpy as np, pandas as pd

fact285_parts = []
for k in [27, 29, 30, 31]:
    orig = pd.read_csv(f"/work/fwd_{k}/factual_original.csv")
    edit = pd.read_csv(f"/work/fwd_{k}/factual_edited.csv")
    assert len(orig) == len(edit)
    for c in ["stay_id", "hour", "event", "observed"]:
        assert orig[c].equals(edit[c]), (k, c)
    paired = orig[["stay_id", "hour", "event"]].copy()
    paired["delta_nll"] = edit.nll.to_numpy() - orig.nll.to_numpy()
    paired["delta_mean"] = edit.predicted_mean.to_numpy() - orig.predicted_mean.to_numpy()
    fact285_parts.append(paired)
    print("batch forward", k, "readings", len(paired),
          "delta", float(paired.delta_nll.mean()))

fact285 = pd.concat(fact285_parts, ignore_index=True)
stay285 = fact285.groupby("stay_id").delta_nll.agg(["sum", "count"])
rng285 = np.random.default_rng(285)
draw285 = rng285.integers(0, len(stay285), size=(5000, len(stay285)))
boot285 = (stay285["sum"].to_numpy()[draw285].sum(axis=1) /
           stay285["count"].to_numpy()[draw285].sum(axis=1))
print("POOLED", len(fact285), "readings", len(stay285), "stays",
      "delta", float(fact285.delta_nll.mean()),
      "stay-bootstrap 95% interval", np.quantile(boot285, [.025, .975]))
print("BY EVENT")
print(fact285.groupby("event")[["delta_nll", "delta_mean"]].mean().to_string())

origB285 = np.asarray(load_original("core.b_in.weight"), dtype=np.float32)
safeB285 = np.asarray(load("core.b_in.weight"), dtype=np.float32)
with np.load("/work/map_regularized_trial.npz") as arc:
    glucoseB285 = arc["baseline"].copy()
    assert np.max(np.abs(safeB285 - arc["core_b_in"])) < 1e-5

G285 = glucoseB285 - origB285
M285 = safeB285 - glucoseB285
np.savez("/work/validated_current400.npz",
         core_b_in=safeB285, original=origB285,
         glucose_correction=G285, map_correction=M285)

candidateB285 = origB285 + (1.08 / 1.15) * G285 + 1.6 * M285
assert candidateB285.shape == origB285.shape
assert np.isfinite(candidateB285).all()
np.savez("/work/balanced_candidate.npz",
         core_b_in=candidateB285, glucose_strength=np.array(1.08),
         map_relative_strength=np.array(1.6))
save_edit("core.b_in.weight", candidateB285.astype(np.float32))
print("Submitted balanced candidate; MAP component norms:",
      float(np.linalg.norm(M285)), float(np.linalg.norm(1.6*M285)))
print("Edited tensors:", edited())

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 285: pooled validation and balanced strength trial\n"
        f"- Previous candidate passed all 400 tuning stays: paired value-NLL delta {fact285.delta_nll.mean():.9f} "
        f"over {len(fact285)} readings; stay-bootstrap 95% interval {np.quantile(boot285,[.025,.975]).tolist()}.\n"
        "- First-batch record convolution targets: insulin x2 approximately -10.37 to -11.39; NE x2 approximately +4.21 to +4.25.\n"
        "- Archived eligible 400-stay fallback in validated_current400.npz.\n"
        "- Submitted glucose strength 1.08 instead of 1.15 and 1.6 times the current MAP addition (full-route strength approximately 0.8).\n"
        "- Trial needs official factual and dose-rollout validation. No gradients or weight optimizer.\n"
        "- Python runs: 285; forward passes: 31.\n"
    )
```