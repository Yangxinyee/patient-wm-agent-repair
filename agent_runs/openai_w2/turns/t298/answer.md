The final cohort’s increase is only about 0.00102, putting the approximate full-tuning increase near 0.00375—below the limit. I’ll calculate the exact paired result and stay-level uncertainty, then recover the matched rollout cohort. This would support the candidate if the factual margin is credible and its dose responses improve toward the record-based targets.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import json
import re

def _r240_paired_factual(forwards):
    frames = []
    keys = ["stay_id", "hour", "event", "observed"]
    for k in forwards:
        a = pd.read_csv(f"/work/fwd_{k}/factual_original.csv")
        b = pd.read_csv(f"/work/fwd_{k}/factual_edited.csv")
        a["_occ"] = a.groupby(keys, dropna=False).cumcount()
        b["_occ"] = b.groupby(keys, dropna=False).cumcount()
        m = a.merge(b, on=keys + ["_occ"], suffixes=("_original", "_edited"),
                    validate="one_to_one")
        assert len(m) == len(a) == len(b)
        m["delta"] = m["nll_edited"] - m["nll_original"]
        m["forward"] = k
        frames.append(m)
    return pd.concat(frames, ignore_index=True)

def _r240_summary(df):
    delta = float(df["delta"].mean())
    clusters = df.groupby("stay_id")["delta"].agg(["sum", "size"])
    residual = clusters["sum"].to_numpy() - delta * clusters["size"].to_numpy()
    g, n = len(clusters), len(df)
    se = float(np.sqrt(g / (g - 1) * np.sum(residual ** 2)) / n)
    return {
        "stays": g, "readings": n,
        "original": float(df["nll_original"].mean()),
        "edited": float(df["nll_edited"].mean()),
        "delta": delta, "cluster_se": se,
        "upper_95_one_sided": delta + 1.6448536269514722 * se
    }

_final_gain_factual = _r240_paired_factual([44, 45, 46, 47])
_final_fallback_factual = _r240_paired_factual([34, 35, 36, 37])
assert _final_gain_factual["stay_id"].nunique() == 400
assert len(_final_gain_factual) == 18977
_final_gain_factual_summary = _r240_summary(_final_gain_factual)
_final_fallback_factual_summary = _r240_summary(_final_fallback_factual)

print("CURRENT CANDIDATE:", json.dumps(_final_gain_factual_summary))
print("ACCEPTED FALLBACK:", json.dumps(_final_fallback_factual_summary))
print("CURRENT BY EVENT:",
      _final_gain_factual.groupby("event")["delta"].agg(["size", "mean"]).to_string())

print("\nROLLOUT FILE RECOVERY:")
for _r240_k in (38, 43):
    _r240_path = Path(f"/work/fwd_{_r240_k}/dose_edit_edited.csv")
    _r240_dose = pd.read_csv(_r240_path)
    print("forward", _r240_k, "shape", _r240_dose.shape,
          "columns", list(_r240_dose.columns))
    print(_r240_dose.head(2).to_dict("records"))
    _r240_staycol = next(c for c in _r240_dose.columns if "stay" in c.lower())
    _r240_ids = sorted(int(x) for x in _r240_dose[_r240_staycol].dropna().unique())
    if _r240_k == 43:
        _final_rollout_ids = _r240_ids
print("MATCHED_ROLLOUT_IDS =", json.dumps(_final_rollout_ids))

_r240_notes = Path("/work/notes.md").read_text()
print("\nSELECTED RECORD-EVIDENCE NOTES:")
_r240_matches = []
for _r240_line in _r240_notes.splitlines():
    if re.search(r"(slope|persistence|latent|target|identif)", _r240_line, re.I):
        if re.search(r"\d", _r240_line):
            _r240_matches.append(_r240_line[:350])
for _r240_line in _r240_matches[-12:]:
    print(_r240_line)

with Path("/work/notes.md").open("a") as _r240_f:
    _r240_f.write(
        "\n\n## Run240: complete final gain-only factual validation\n"
        "Current candidate_B237 unchanged; only core.b_in.weight edited.\n"
        "Forwards44–47: " + json.dumps(_final_gain_factual_summary) + "\n"
        "Accepted fallback forwards34–37: "
        + json.dumps(_final_fallback_factual_summary) + "\n"
        "Recovered matched rollout stay IDs. Candidate remains provisional "
        "pending exact dose-rollout comparison; estimates from the records "
        "remain assumption-dependent, not revealed simulator truth.\n"
    )
```