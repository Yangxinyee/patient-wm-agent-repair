The final cohort’s increase is only **0.00137 nats**. All four cohorts now pass individually, and their pooled increase is about **0.00331**. I’ll record the full audit and recover the established rollout targets before testing this exact checkpoint. Agreement with those record-derived targets—not just the correct signs—will determine whether to keep it.

```python
import numpy as np
import pandas as pd
from pathlib import Path
import re

_repair_pairs = []
for _repair_k in (39, 40, 41, 42):
    _repair_o = pd.read_csv(f"/work/fwd_{_repair_k}/factual_original.csv")
    _repair_e = pd.read_csv(f"/work/fwd_{_repair_k}/factual_edited.csv")
    assert _repair_o[["stay_id", "hour", "event", "observed"]].equals(
        _repair_e[["stay_id", "hour", "event", "observed"]]
    )
    _repair_p = _repair_o[["stay_id", "event"]].copy()
    _repair_p["original_nll"] = _repair_o.nll
    _repair_p["edited_nll"] = _repair_e.nll
    _repair_p["delta"] = _repair_e.nll - _repair_o.nll
    _repair_p["forward"] = _repair_k
    _repair_pairs.append(_repair_p)

_repair_all400 = pd.concat(_repair_pairs, ignore_index=True)
_repair_mu = _repair_all400.delta.mean()
_repair_g = _repair_all400.groupby("stay_id").delta.agg(["sum", "count"])
_repair_se = np.sqrt(
    len(_repair_g) / (len(_repair_g) - 1)
    * np.square(_repair_g["sum"] - _repair_mu * _repair_g["count"]).sum()
) / len(_repair_all400)
print("=== FACTUAL AUDIT: ALL 400 TUNING STAYS ===")
print(_repair_all400.groupby("forward").delta.agg(["count", "mean"]))
print(_repair_all400.groupby("event").delta.agg(["count", "mean"]))
print({
    "n_stays": len(_repair_g),
    "original_nll": _repair_all400.original_nll.mean(),
    "edited_nll": _repair_all400.edited_nll.mean(),
    "delta": _repair_mu,
    "stay_cluster_SE": _repair_se,
    "95pct_upper": _repair_mu + 1.96 * _repair_se
})

print("\n=== ESTABLISHED TARGETS AND VALIDATED EDIT DESIGN ===")
_repair_sections = re.split(
    r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.MULTILINE
)
for _repair_prefix in ("## Run 7\n", "## Forward 20", "## Run 234",
                       "## Forward 29"):
    for _repair_s in _repair_sections:
        if _repair_s.startswith(_repair_prefix):
            print(_repair_s[:1500])
            break

print("\n=== EFFECT/TARGET VARIABLE NAMES ===")
print(", ".join(sorted(
    k for k in list(globals())
    if not k.startswith("_")
    and any(t in k.lower() for t in ("kernel", "target", "impulse", "dml", "effect"))
))[:1800])

print("\n=== CHECKPOINT REPRODUCTION ===")
with np.load("/work/candidate90_35_bin80x4.npz") as _repair_checkpoint:
    for _repair_name in _repair_checkpoint.files:
        if _repair_name in NAMES:
            _repair_tensor = load(_repair_name)
            if hasattr(_repair_tensor, "numpy"):
                _repair_tensor = _repair_tensor.numpy()
            print(_repair_name,
                  "max_difference",
                  float(np.max(np.abs(
                      _repair_tensor - _repair_checkpoint[_repair_name]
                  ))))

with Path("/work/notes.md").open("a") as _repair_f:
    _repair_f.write(
        "\n## Forward 41–42 / Run 269: complete factual validation\n"
        "No weights changed. Current checkpoint is insulin .90 / NE .35 plus "
        "shrink80-scale4 previous-bin cancellation (forward hash 864804d79e).\n"
        + _repair_all400.groupby("forward").delta.agg(["count", "mean"]).to_string()
        + f"\nAll400 pooled original NLL {_repair_all400.original_nll.mean():.8f}, "
        f"edited {_repair_all400.edited_nll.mean():.8f}, delta {_repair_mu:.8f}; "
        f"stay-cluster SE {_repair_se:.8f}, upper95 {_repair_mu+1.96*_repair_se:.8f}. "
        "All four 100-stay cohorts individually pass the .005 threshold. "
        "Rollout validation of this exact checkpoint is still pending.\n"
    )
```