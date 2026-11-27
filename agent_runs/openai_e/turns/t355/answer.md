C also passes the second factual check, with an increase of about 0.00331 nats. Both tested cohorts are below the limit. I’ll calculate exact paired totals and verify the archive before spending the last forward pass on the rollout. Consistent reductions relative to B would support keeping C; an alignment error would invalidate that comparison.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import hashlib

def paired292(k):
    o = pd.read_csv(f"/work/fwd_{k}/factual_original.csv")
    e = pd.read_csv(f"/work/fwd_{k}/factual_edited.csv")
    ncols = [c for c in o.columns if "nll" in c.lower()]
    nc = "nll" if "nll" in ncols else ncols[0]
    assert len(ncols) == 1 or nc == "nll", ncols
    assert len(o) == len(e)
    for c in ["stay_id", "hour", "minute", "time", "event", "code",
              "code_name", "value", "true_value", "true_bin"]:
        if c in o.columns:
            assert o[c].equals(e[c]), (k, c)
    assert np.isfinite(o[nc]).all() and np.isfinite(e[nc]).all()
    ec = next((c for c in ["event", "code_name", "code", "event_type"]
               if c in o.columns), None)
    q = pd.DataFrame({
        "stay_id": o["stay_id"],
        "original_nll": o[nc],
        "edited_nll": e[nc],
        "delta": e[nc] - o[nc],
    })
    if ec is not None:
        q["event"] = o[ec]
    return q, o.columns.tolist()

pairs292 = {}
for kc, kb in [(48, 43), (49, 44)]:
    c, columns292 = paired292(kc)
    b, _ = paired292(kb)
    assert c["stay_id"].equals(b["stay_id"])
    assert np.max(np.abs(c.original_nll - b.original_nll)) < 1e-7
    c["B_delta"] = b["delta"].to_numpy()
    pairs292[kc] = c
    print("C forward", kc, {
        "n_readings": len(c),
        "original_nll": float(c.original_nll.mean()),
        "C_nll": float(c.edited_nll.mean()),
        "C_delta": float(c.delta.mean()),
        "B_delta_same_cohort": float(c.B_delta.mean()),
        "C_to_B_mean_cost_ratio": float(c.delta.mean() / c.B_delta.mean()),
    })

pool292 = pd.concat(list(pairs292.values()), ignore_index=True)
print("\nColumns:", columns292)
print("C pooled observed:", {
    "n_stays": int(pool292.stay_id.nunique()),
    "n_readings": len(pool292),
    "original_nll": float(pool292.original_nll.mean()),
    "edited_nll": float(pool292.edited_nll.mean()),
    "delta": float(pool292.delta.mean()),
})
if "event" in pool292:
    print("C by event:\n", pool292.groupby("event").agg(
        n=("delta", "size"), nll_increase=("delta", "mean")).to_string())

clusters292 = pool292.groupby("stay_id").agg(
    cost=("delta", "sum"), n=("delta", "size"))
rng292 = np.random.default_rng(20261110)
ix292 = rng292.integers(0, len(clusters292), size=(20000, len(clusters292)))
boot292 = (clusters292.cost.to_numpy()[ix292].sum(axis=1) /
           clusters292.n.to_numpy()[ix292].sum(axis=1))
print("Selected-200 stay-cluster bootstrap:", {
    "95_percent_interval": np.quantile(boot292, [.025, .975]).tolist(),
    "one_sided_95_upper": float(np.quantile(boot292, .95)),
})
print("These 200 were selected as B's highest-cost cohorts; "
      "this is not an unbiased all400 estimate for C.")

bpool292 = pd.concat([paired292(k)[0] for k in [43, 44, 45, 46]],
                     ignore_index=True)
ratio292 = float(pool292.delta.mean() / pool292.B_delta.mean())
print("B all400 measured delta:", float(bpool292.delta.mean()))
print("C all400 scalar-shrink extrapolation (NOT measured):",
      float(bpool292.delta.mean() * ratio292))

with np.load("/work/joint291.npz", allow_pickle=False) as z:
    archive292 = z["delta"].copy()
weight292 = np.asarray(load("core.b_in.weight"), dtype=np.float32)
original292 = np.asarray(load_original("core.b_in.weight"), dtype=np.float32)
assert edited() == ["core.b_in.weight"]
assert np.max(np.abs(weight292 - (original292 + archive292))) < 1e-5
hash292 = hashlib.sha256(weight292.tobytes()).hexdigest()
print("Verified current C weight SHA256:", hash292)

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 292: final candidate C paired factual audit\n"
        f"Forward48 delta={pairs292[48].delta.mean():.10f}; "
        f"forward49 delta={pairs292[49].delta.mean():.10f}. "
        f"Pooled observed C delta={pool292.delta.mean():.10f} over "
        f"{len(pool292)} readings / {pool292.stay_id.nunique()} stays. "
        "Both cohorts pass .005; they were B's two highest-cost cohorts, "
        "not a random population sample. "
        f"Selected-200 bootstrap95={np.quantile(boot292,[.025,.975]).tolist()}. "
        f"Cost ratio C/B={ratio292:.6f}; scalar extrapolation to all400 is "
        f"{bpool292.delta.mean()*ratio292:.10f}, explicitly not measured. "
        f"Only core.b_in.weight edited; archive verified; SHA256={hash292}. "
        "No weights changed. Last remaining forward reserved for exact C rollout "
        "on tuning100..199, directly comparable to B forward47.\n"
        "Python runs: 292; forwards: 49.\n"
    )
```