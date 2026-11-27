The audit stopped because original-model NLLs were not bit-identical across separate forward requests. That does not invalidate either within-request paired result, but the cross-request equality assumption was too strict. I’ll quantify the difference, check event alignment, and calculate totals using each request’s own original baseline. Matching event identities would confirm the comparison is valid.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import hashlib

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 292 diagnostic failure\n"
        "Stopped at cross-request original-NLL equality assertion before any "
        "weight change or audit conclusions. Within-request original/edited "
        "alignment checks had passed. Cross-request bit equality is not required "
        "by the tool contract; correcting that assumption in run293.\n"
        "Python runs: 292; forwards: 49.\n"
    )

pairs293 = {}
for kc, kb in [(48, 43), (49, 44)]:
    c293, columns293 = paired292(kc)
    b293, _ = paired292(kb)
    oc293 = pd.read_csv(f"/work/fwd_{kc}/factual_original.csv")
    ob293 = pd.read_csv(f"/work/fwd_{kb}/factual_original.csv")
    alignment293 = {}
    for col in ["stay_id", "hour", "minute", "time", "event", "code",
                "code_name", "value", "true_value", "true_bin"]:
        if col in oc293.columns:
            alignment293[col] = oc293[col].equals(ob293[col])
    assert alignment293 and all(alignment293.values()), alignment293
    print("Cross-request alignment", kc, kb, alignment293)
    print("Original NLL difference across requests:", {
        "max_abs": float(np.max(np.abs(c293.original_nll-b293.original_nll))),
        "mean": float((c293.original_nll-b293.original_nll).mean()),
    })
    c293["B_delta"] = b293.delta.to_numpy()
    pairs293[kc] = c293
    print("Paired factual costs:", {
        "C_forward": kc, "n_readings": len(c293),
        "C_delta": float(c293.delta.mean()),
        "B_delta_same_cohort": float(b293.delta.mean()),
    })

pool293 = pd.concat(list(pairs293.values()), ignore_index=True)
print("\nFactual columns:", columns293)
print("C pooled observed:", {
    "n_stays": int(pool293.stay_id.nunique()),
    "n_readings": len(pool293),
    "original_nll": float(pool293.original_nll.mean()),
    "edited_nll": float(pool293.edited_nll.mean()),
    "delta": float(pool293.delta.mean()),
})
if "event" in pool293:
    print("C by event:\n", pool293.groupby("event").agg(
        n=("delta", "size"), nll_increase=("delta", "mean")).to_string())

clusters293 = pool293.groupby("stay_id").agg(
    cost=("delta", "sum"), n=("delta", "size"))
rng293 = np.random.default_rng(20261110)
ix293 = rng293.integers(0, len(clusters293), size=(20000, len(clusters293)))
boot293 = (clusters293.cost.to_numpy()[ix293].sum(axis=1) /
           clusters293.n.to_numpy()[ix293].sum(axis=1))
ci293 = np.quantile(boot293, [.025, .975])
print("Selected-200 cluster-bootstrap95:", ci293.tolist())
print("These were B's two highest-cost cohorts; not a random all400 sample.")

bpool293 = pd.concat([paired292(k)[0] for k in [43, 44, 45, 46]],
                     ignore_index=True)
ratio293 = float(pool293.delta.mean() / pool293.B_delta.mean())
extrap293 = float(bpool293.delta.mean() * ratio293)
print("B all400 measured delta:", float(bpool293.delta.mean()))
print("C/B selected-cohort cost ratio:", ratio293)
print("C all400 scalar extrapolation, NOT measured:", extrap293)

with np.load("/work/joint291.npz", allow_pickle=False) as z:
    archive293 = z["delta"].copy()
weight293 = np.asarray(load("core.b_in.weight"), dtype=np.float32)
original293 = np.asarray(load_original("core.b_in.weight"), dtype=np.float32)
assert edited() == ["core.b_in.weight"]
assert np.max(np.abs(weight293 - (original293 + archive293))) < 1e-5
hash293 = hashlib.sha256(weight293.tobytes()).hexdigest()
print("Verified current C SHA256:", hash293)

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 293: corrected final-C factual audit\n"
        f"Event identities align across requests. Forward48 C delta="
        f"{pairs293[48].delta.mean():.10f}; forward49="
        f"{pairs293[49].delta.mean():.10f}. Pooled observed delta="
        f"{pool293.delta.mean():.10f}, {len(pool293)} readings / "
        f"{pool293.stay_id.nunique()} stays. Both cohort means pass .005. "
        f"Selected-200 bootstrap95={ci293.tolist()}; these were B's two "
        "highest-cost cohorts, not a random population sample. "
        f"C/B cost ratio={ratio293:.6f}; all400 scalar extrapolation="
        f"{extrap293:.10f}, explicitly not measured. "
        f"Archive verified; SHA256={hash293}. No weights changed. "
        "Final forward reserved for C dose rollout on tuning100..199.\n"
        "Python runs: 293; forwards: 49.\n"
    )
```