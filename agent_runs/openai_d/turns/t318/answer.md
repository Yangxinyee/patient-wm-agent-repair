The second cohort’s factual NLL increase is **0.00259 nats**, comfortably below the limit. Together, the first 200 stays give a pooled increase of about **0.00359**. I’ll log the exact paired results and recover the later dose-response calibration before testing the remaining cohorts. The candidate is promising, but its rollout responses still need verification.

```python
import numpy as np
import pandas as pd
import re
from pathlib import Path

_repair_pairs = []
for _repair_k in (39, 40):
    _repair_o = pd.read_csv(f"/work/fwd_{_repair_k}/factual_original.csv")
    _repair_e = pd.read_csv(f"/work/fwd_{_repair_k}/factual_edited.csv")
    _repair_keys = ["stay_id", "hour", "event", "observed"]
    assert _repair_o[_repair_keys].equals(_repair_e[_repair_keys])
    _repair_p = _repair_o[["stay_id", "event"]].copy()
    _repair_p["delta"] = _repair_e.nll - _repair_o.nll
    _repair_p["forward"] = _repair_k
    _repair_pairs.append(_repair_p)

_repair_pooled = pd.concat(_repair_pairs, ignore_index=True)
_repair_mu = _repair_pooled.delta.mean()
_repair_g = _repair_pooled.groupby("stay_id").delta.agg(["sum", "count"])
_repair_se = np.sqrt(
    len(_repair_g) / (len(_repair_g) - 1)
    * np.square(_repair_g["sum"] - _repair_mu * _repair_g["count"]).sum()
) / len(_repair_pooled)
print("=== EXACT FACTUAL VALIDATION ===")
print(_repair_pooled.groupby("forward").delta.agg(["count", "mean"]))
print({"pooled_delta": _repair_mu, "stay_cluster_SE": _repair_se,
       "95pct_upper": _repair_mu + 1.96 * _repair_se})

print("\n=== RELEVANT LATER NOTEBOOK SECTIONS ===")
_repair_sections = re.split(r"(?=^## )", Path("/work/notes.md").read_text(),
                            flags=re.MULTILINE)
_repair_matched = [
    s for s in _repair_sections
    if any(t in s.lower() for t in
           ("causal target", "rollout target", "record-derived",
            "true effect", "structural", "identification"))
]
print("Matching headings:",
      [s.splitlines()[0] for s in _repair_matched])
for _repair_s in _repair_matched[-3:]:
    print(_repair_s[:1300])

print("\n=== AVAILABLE CUSTOM FUNCTIONS ===")
print(", ".join(sorted(
    k for k, v in list(globals().items())
    if callable(v) and getattr(v, "__module__", "") == "__main__"
)))

print("\n=== RECENT CALIBRATION VARIABLE NAMES ===")
print(", ".join(sorted(
    k for k in list(globals())
    if not k.startswith("_") and re.search(r"(25[5-9]|26[0-5])", k)
))[:1600])

print("\n=== LATEST SAVED DOSE-ROLLOUT FILES ===")
_repair_dose_files = sorted(
    Path("/work").glob("fwd_*/dose_edit_edited.csv"),
    key=lambda p: int(p.parent.name.split("_")[1])
)
for _repair_p in _repair_dose_files[-2:]:
    _repair_d = pd.read_csv(_repair_p)
    print(str(_repair_p), _repair_d.shape, _repair_d.columns.tolist())
    print(_repair_d.head(2).to_string(index=False))

with Path("/work/notes.md").open("a") as _repair_f:
    _repair_f.write(
        "\n## Runs 266–268 / Forwards 39–40: reduced candidate validation\n"
        "Recovered architecture and early causal robustness checks; no weights changed. "
        "Current insulin .90 / NE .35 with shrink80-scale4 previous-bin cancellation "
        "is unchanged (forward hash 864804d79e).\n"
        + _repair_pooled.groupby("forward").delta.agg(["count", "mean"]).to_string()
        + f"\nReading-weighted pooled delta on first200 tuning stays: {_repair_mu:.8f}; "
        f"stay-cluster SE {_repair_se:.8f}, upper95 {_repair_mu + 1.96*_repair_se:.8f}. "
        "Need remaining-cohort factual validation and exact candidate rollout checks.\n"
    )
```