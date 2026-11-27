The independent rollout confirms stronger insulin responses, but all effects remain much too small. I’ll keep the tested edits unchanged, pool both rollout cohorts using the tool’s stay-hour weighting, and record the final audit. Agreement across cohorts supports a partial correction—not a complete repair.

```python
import json
from pathlib import Path
import numpy as np
import pandas as pd

final_ops = ops()
assert len(final_ops) == 2
assert final_ops[0]["name"] == "embed.numeric.weight"
assert final_ops[0]["side"] == "in" and final_ops[0]["alpha"] == 3.0
assert final_ops[1]["name"] == "core.b_in.weight"
assert np.isclose(final_ops[1]["c"], 1.30)
assert np.allclose(
    np.load("/work/" + final_ops[0]["vector"]),
    np.array([0., 0., 1., 0., 1., 0.]) / np.sqrt(2)
)
for k in (47, 48, 49, 50):
    summary = json.loads(Path(f"/work/fwd_{k}/summary.json").read_text())
    assert summary["edit_hash"] == "1fb70dfa56"

final_rollout = {}
for model in ("original", "edited"):
    pieces = []
    cohort_ids = []
    for k in (47, 50):
        df = pd.read_csv(f"/work/fwd_{k}/dose_edit_{model}.csv")
        cohort_ids.append(set(df["stay_id"]))
        pieces.append(df.loc[df["reached"] == True].copy())
    assert cohort_ids[0].isdisjoint(cohort_ids[1])
    pooled = pd.concat(pieces, ignore_index=True)
    final_rollout[model] = {}
    for edit, rows in pooled.groupby("edit"):
        clusters = rows.groupby("stay_id")["shift"].agg(["sum", "count"])
        mean = rows["shift"].mean()
        residual = clusters["sum"] - mean * clusters["count"]
        se = (np.sqrt(len(clusters) / (len(clusters) - 1) *
                      np.square(residual).sum()) / clusters["count"].sum())
        final_rollout[model][edit] = {
            "mean_shift": float(mean),
            "stay_cluster_se": float(se),
            "n_stay_hours": len(rows),
            "n_stays": len(clusters)
        }

final_audit = {
    "operations": final_ops,
    "edit_hash": "1fb70dfa56",
    "factual": audit_130_final,
    "rollouts": final_rollout,
    "training_record_effect_estimates": target_train,
    "forward_passes_used": 50,
    "python_runs_used": 119,
    "assessment": (
        "Partial repair. Insulin sensitivity strengthened reproducibly; "
        "both drug responses remain substantially below record-based estimates. "
        "Factual loss passes the observed budget, with no guarantee on unseen stays."
    )
}
print(json.dumps(final_audit, indent=2))
Path("/work/final_audit.json").write_text(json.dumps(final_audit, indent=2))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Python run 119 / forward 50: final frozen submission\n"
        "- Kept numeric joint dose/rate input gain 4 and recurrent input gain 1.30;"
        " no untested final edit.\n"
        "- Verified exact-candidate operation list and matching edit hashes.\n"
        f"- Factual audit, 300 disjoint stays: {audit_130_final}\n"
        f"- Pooled reached-stay-hour rollout audit, 200 disjoint stays: {final_rollout}\n"
        f"- Training-record effect estimates: {target_train}\n"
        "- Earlier equal-stay summaries were diagnostics; final forecasts use"
        " the tool's actual reached-stay-hour scoring mean.\n"
        "- This is a partial improvement, not successful recovery of the full"
        " record-estimated effects; NE halving is not improved versus original.\n"
        "- Forward passes used: 50/50. Python runs used: 119/300."
        " All final operations remain unchanged.\n"
    )
```
