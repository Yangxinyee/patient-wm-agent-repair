The archives store weight changes, not complete weights, which explains the nonzero distances printed above. I’ll verify B against the original weights plus its saved change, and recompute the fallback’s pooled factual loss. Exact agreement will establish a reliable baseline for the remaining tests.

```python
import numpy as np
import pandas as pd
from pathlib import Path

for filename in ["joint236.npz", "joint274.npz", "joint284.npz"]:
    with np.load("/work/" + filename) as z:
        expected = original_B + z["delta"]
        reconstruction = (
            float(z["insulin_scale"]) * z["insulin_delta"]
            + float(z["ne_scale"]) * z["ne_delta"]
        )
        print(filename,
              "installed_max_abs_error",
              float(np.max(np.abs(current_B - expected))),
              "component_reconstruction_error",
              float(np.linalg.norm(z["delta"] - reconstruction)),
              "delta_norm", float(np.linalg.norm(z["delta"])))

fallback_pairs = []
keys = ["stay_id", "hour", "event", "observed"]
for k in [36, 37, 38, 39]:
    fo = pd.read_csv(f"/work/fwd_{k}/factual_original.csv")
    fe = pd.read_csv(f"/work/fwd_{k}/factual_edited.csv")
    assert fo[keys].equals(fe[keys]), f"Unpaired factual rows in forward {k}"
    part = fo[["stay_id", "event", "nll"]].copy()
    part["edited_nll"] = fe["nll"].to_numpy()
    part["delta"] = part["edited_nll"] - part["nll"]
    fallback_pairs.append(part)
    print("fallback cohort", k, "readings", len(part),
          "NLL increase", float(part["delta"].mean()))

fallback_all = pd.concat(fallback_pairs, ignore_index=True)
print("FALLBACK ALL400",
      "stays", fallback_all["stay_id"].nunique(),
      "readings", len(fallback_all),
      "original_NLL", float(fallback_all["nll"].mean()),
      "edited_NLL", float(fallback_all["edited_nll"].mean()),
      "increase", float(fallback_all["delta"].mean()))
print("FALLBACK BY EVENT",
      fallback_all.groupby("event")["delta"].mean().to_dict())

clusters = fallback_all.groupby("stay_id")["delta"].agg(["sum", "count"])
rng_audit = np.random.default_rng(287)
draws = rng_audit.integers(0, len(clusters), size=(10000, len(clusters)))
boot_delta = (
    clusters["sum"].to_numpy()[draws].sum(axis=1)
    / clusters["count"].to_numpy()[draws].sum(axis=1)
)
print("FALLBACK cluster-bootstrap 95%",
      np.quantile(boot_delta, [0.025, 0.975]).tolist())

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 287: verify installed candidate and pooled fallback\n"
        "Checked original-plus-archived-delta reconstruction and component conventions. "
        "Recomputed joint274 paired factual fit across all400 tuning stays, including "
        "stay-cluster bootstrap. No weights changed. Candidate B remains unaccepted "
        "pending all400 factual validation and paired rollout.\n"
        "Python runs: 287; forwards: 42.\n"
    )
```