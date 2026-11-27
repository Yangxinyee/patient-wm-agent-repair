The edit improves all four probe responses relative to the original, although they remain much smaller than the record-based estimates. Insulin doubling now gives −3.00 versus roughly −12 estimated from the records. I’ll audit factual-loss uncertainty and earlier safe candidates before validating on new stays. A loss increase near or above 0.005 would argue for reducing the gain.

```python
from pathlib import Path
import numpy as np
import pandas as pd

orig43 = pd.read_csv("/work/fwd_43/factual_original.csv")
edit43 = pd.read_csv("/work/fwd_43/factual_edited.csv")
keys43 = ["stay_id", "hour", "event", "observed"]
assert orig43[keys43].equals(edit43[keys43])

paired43 = orig43[["stay_id", "event"]].copy()
paired43["delta_nll"] = edit43["nll"].to_numpy() - orig43["nll"].to_numpy()
delta43 = paired43["delta_nll"].mean()
clusters43 = paired43.groupby("stay_id")["delta_nll"].agg(["sum", "count"])
g43, n43 = len(clusters43), len(paired43)
se43 = np.sqrt(
    g43 / (g43 - 1)
    * np.square(clusters43["sum"] - delta43 * clusters43["count"]).sum()
    / n43**2
)
print("=== CURRENT FACTUAL AUDIT ===")
print({"delta_nll": float(delta43), "stay_cluster_se": float(se43),
       "approx_95pct_interval": [float(delta43 - 1.96*se43),
                                float(delta43 + 1.96*se43)]})
print(paired43.groupby("event")["delta_nll"].agg(["mean", "count"]))

print("\n=== EARLIER FACTUAL RESULTS ===")
for k in range(1, 44):
    po = Path(f"/work/fwd_{k}/factual_original.csv")
    pe = Path(f"/work/fwd_{k}/factual_edited.csv")
    if po.exists() and pe.exists():
        oo, ee = pd.read_csv(po), pd.read_csv(pe)
        print(k, "stays", ee["stay_id"].nunique(), "n", len(ee),
              "original", round(oo["nll"].mean(), 6),
              "edited", round(ee["nll"].mean(), 6),
              "delta", round(ee["nll"].mean()-oo["nll"].mean(), 6))

text112 = Path("/work/notes.md").read_text()
print("\n=== RECORD-BASED EFFECT AND NUMERIC-GAIN NOTES ===")
selected112 = []
for line in text112.splitlines():
    low = line.lower()
    if any(s in low for s in [
        "record-based", "probe targets", "target shifts", "causal estimate",
        "numeric gain", "numeric-gain", "numeric_gain",
        "persistence", "local projection", "rho=", "dose innovation"
    ]):
        selected112.append(line[:1100])
print("\n".join(selected112)[:5500])

print("\nCurrent operations:", ops())
with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Python run 112 / forwards 43–44: recurrent gain 1.35\n"
        f"- Probe factual NLL increase {delta43:.6f}; stay-cluster SE {se43:.6f}.\n"
        "- Dose readouts: insulin x2 -2.997, half +2.128; NE x2 +0.127, half -0.061.\n"
        "- All move toward record-based effects versus the original, but remain substantially too small.\n"
        "- Retained both operations pending independent factual validation.\n"
        "- Forward passes used: 44; 6 remain.\n"
    )
```