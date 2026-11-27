The final cohort’s increase is about 0.00215, putting B’s all-400 average near 0.00365—below the limit. I’ll calculate the exact paired result and stay-level uncertainty before testing dose responses. A wide uncertainty interval would still argue for caution despite the passing average.

```python
import numpy as np
import pandas as pd
from pathlib import Path

candidate_B_parts = []
pair_keys = ["stay_id", "hour", "event", "observed"]
candidate_B_cohort_costs = {}

for k in [43, 44, 45, 46]:
    fo = pd.read_csv(f"/work/fwd_{k}/factual_original.csv")
    fe = pd.read_csv(f"/work/fwd_{k}/factual_edited.csv")
    assert fo[pair_keys].equals(fe[pair_keys]), f"Pairing failure in forward {k}"
    p = fo[["stay_id", "event", "nll"]].copy()
    p["edited_nll"] = fe["nll"].to_numpy()
    p["delta"] = p["edited_nll"] - p["nll"]
    candidate_B_parts.append(p)
    candidate_B_cohort_costs[k] = float(p["delta"].mean())
    print("B cohort", k, "readings", len(p),
          "NLL increase", candidate_B_cohort_costs[k])

candidate_B_all = pd.concat(candidate_B_parts, ignore_index=True)
assert candidate_B_all["stay_id"].nunique() == 400
assert len(candidate_B_all) == 18654
candidate_B_cost = float(candidate_B_all["delta"].mean())
print("B ALL400",
      "original_NLL", float(candidate_B_all["nll"].mean()),
      "edited_NLL", float(candidate_B_all["edited_nll"].mean()),
      "increase", candidate_B_cost,
      "headroom", 0.005 - candidate_B_cost)
print("B BY EVENT",
      candidate_B_all.groupby("event")["delta"].mean().to_dict())

B_clusters = candidate_B_all.groupby("stay_id")["delta"].agg(["sum", "count"])
rng_B = np.random.default_rng(288)
B_draws = rng_B.integers(0, len(B_clusters), size=(20000, len(B_clusters)))
B_boot = (
    B_clusters["sum"].to_numpy()[B_draws].sum(axis=1)
    / B_clusters["count"].to_numpy()[B_draws].sum(axis=1)
)
B_interval = np.quantile(B_boot, [0.025, 0.975])
print("B cluster-bootstrap 95%", B_interval.tolist())
print("B one-sided 95% upper", float(np.quantile(B_boot, 0.95)))
print("Bootstrap fraction exceeding .005", float(np.mean(B_boot > 0.005)))
print("Fallback all400 increase", float(fallback_all["delta"].mean()))

with np.load("/work/joint284.npz") as z:
    expected = audit_numpy(load_original("core.b_in.weight")) + z["delta"]
    assert np.array_equal(audit_numpy(load("core.b_in.weight")), expected)
print("Confirmed B unchanged; edited tensors:", edited())

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 288: candidate B all400 paired factual audit\n"
        f"B cohort costs={candidate_B_cohort_costs}. All400 NLL increase="
        f"{candidate_B_cost:.10f}; cluster-bootstrap95={B_interval.tolist()}; "
        f"one-sided95 upper={float(np.quantile(B_boot, 0.95)):.10f}. "
        "The pooled observed mean meets .005, but uncertainty must be considered. "
        "No weights changed; candidate B still needs exact dose-rollout validation. "
        "Only core.b_in.weight is edited, exactly original plus joint284 delta.\n"
        "Python runs: 288; forwards: 46.\n"
    )
```