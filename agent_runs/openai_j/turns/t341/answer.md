The fourth group adds +0.00348 nats, so every group passes individually. I’ll now calculate the pooled loss and stay-level uncertainty, then validate rollouts beyond the first 100 stays. A pooled upper bound below 0.005 would support retaining the current weights, although it cannot guarantee unseen-stay performance.

```python
import numpy as np, pandas as pd, json
from pathlib import Path

pairs289 = []
for n289 in [40, 43, 44, 45]:
    o289 = pd.read_csv(f"/work/fwd_{n289}/factual_original.csv")
    e289 = pd.read_csv(f"/work/fwd_{n289}/factual_edited.csv")
    keys289 = ["stay_id", "hour", "event", "observed"]
    assert o289[keys289].equals(e289[keys289]), f"row mismatch in fwd_{n289}"
    z289 = o289[keys289].copy()
    z289["original_nll"] = o289["nll"].to_numpy()
    z289["edited_nll"] = e289["nll"].to_numpy()
    z289["delta"] = z289["edited_nll"] - z289["original_nll"]
    z289["forward_dir"] = n289
    pairs289.append(z289)

audit289 = pd.concat(pairs289, ignore_index=True)
assert audit289["stay_id"].nunique() == 400
assert len(audit289) == 18654
cluster289 = audit289.groupby("stay_id").agg(
    delta_sum=("delta", "sum"), n=("delta", "size"))
rng289 = np.random.default_rng(289)
draw289 = rng289.integers(0, len(cluster289), size=(10000, len(cluster289)))
sums289 = cluster289["delta_sum"].to_numpy()
counts289 = cluster289["n"].to_numpy()
boot289 = sums289[draw289].sum(axis=1) / counts289[draw289].sum(axis=1)
pooled289 = float(audit289["delta"].mean())
ci289 = np.quantile(boot289, [0.025, 0.975])
print("=== CURRENT FOUR-TENSOR FACTUAL AUDIT ===")
print("stays", 400, "readings", len(audit289))
print("original NLL", audit289["original_nll"].mean())
print("edited NLL", audit289["edited_nll"].mean())
print("delta NLL", pooled289)
print("stay-bootstrap SE", boot289.std(ddof=1))
print("95% interval", ci289)
print("95% one-sided upper", np.quantile(boot289, .95))
print("value-loss sum increase", audit289["delta"].sum())
print(audit289.groupby("event").agg(
    n=("delta", "size"), original_nll=("original_nll", "mean"),
    edited_nll=("edited_nll", "mean"), delta=("delta", "mean")).to_string())
audit289.to_csv("/work/final_factual_audit289.csv", index=False)

print("\n=== WEIGHT CHANGE AUDIT ===")
for k289 in edited():
    w289 = load(k289)
    wo289 = load_original(k289)
    if hasattr(w289, "detach"):
        w289 = w289.detach().cpu().numpy()
    if hasattr(wo289, "detach"):
        wo289 = wo289.detach().cpu().numpy()
    dw289 = np.asarray(w289)-np.asarray(wo289)
    print(k289, "changed entries", int(np.count_nonzero(dw289)),
          "L2", float(np.linalg.norm(dw289)), "max", float(np.abs(dw289).max()))

print("\n=== CONSTRUCTION / IDENTIFICATION NOTE EXCERPTS ===")
lines289 = Path("/work/notes.md").read_text().splitlines()
hits289 = [i for i,l in enumerate(lines289)
           if re.search(r"kernel fit|selected kernel|cross.fit|DML|low.rank|orthogon|head177|confound", l, re.I)]
for i289 in hits289[:18]:
    print("\n".join(lines289[max(0,i289-1):min(len(lines289),i289+2)])[:420])

with open("/work/notes.md", "a") as f289:
    f289.write("\n## Run 289: full current-candidate factual validation\n"
               f"- Four current-hash factual groups, 400 stays / {len(audit289)} readings: "
               f"dNLL {pooled289:+.8f}; stay-bootstrap 95% interval [{ci289[0]:.8f}, {ci289[1]:.8f}].\n"
               "- Bootstrap uncertainty is conditional on candidate selection, not a guarantee for evaluation stays.\n"
               "- All non-value heads and all state/action weights remain original; total-output loss was not directly reported.\n"
               "- Current edit unchanged. Counts: 289 Python runs, 43 forward passes.\n")
```