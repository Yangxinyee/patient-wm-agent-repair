Removing the injection reduces the independent NLL increase to just 0.00031 nats, implicating the injection rather than the baseline edits. I’ll test an existing projection fitted with a stronger factual penalty, using larger drug-effect gains. It would support this approach if rollout effects grow while factual loss stays within 0.005; otherwise I’ll reduce or reject it.

```python
import numpy as np, pandas as pd

print("EXISTING PROJECTION DIAGNOSTICS")
for n295 in ("stats286", "metrics286", "moments286"):
    if n295 in globals():
        x295 = globals()[n295]
        if isinstance(x295, dict):
            print(n295, str(x295)[:2100])
        else:
            print(n295, str(x295)[:2100])
print("Projection column norms:")
for key295, val295 in fits286.items():
    print(key295, np.round(np.linalg.norm(val295, axis=0), 5))

P295 = fits286[("feedback4", 100.0)].copy()
gains295 = np.array([-6.0, 2.0])
W295 = (a292(load_original("core.b_in.weight")) +
        B286 @ np.diag(gains295) @ P295.T).astype(np.float32)
assert np.isfinite(W295).all()
save_edit("core.b_in.weight", W295)
candidate295 = {n: a292(load(n)).copy() for n in edited()}
print("Installed candidate295: factual penalty 100, gains", gains295.tolist())
print("Injection Frobenius norm:", float(np.linalg.norm(
    W295 - a292(load_original("core.b_in.weight")))))
print("Immediate feedback matrix:",
      np.round(P295.T @ (d286[:, None] * B286 @ np.diag(gains295)), 7)
      if np.asarray(d286).size == 192 else "d286 shape differs; existing constraints retained")

comparison_ids295 = sorted(int(x) for x in
    pd.read_csv("/work/fwd_32/factual_original.csv")["stay_id"].unique())
print("Matched rollout comparison IDs:", comparison_ids295)
print("Next factual request:")
print({"type": "factual", "model": "both", "split": "tuning", "stays": validation_ids293})

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 295\n"
            "Forward35 safe236 baseline independent factual delta approximately +.00031; "
            "the recurrent injection caused most of Forward34 loss. Matched six-lag estimates "
            "on Forward33 reached rows: insulin x2 -11.20288, half +5.60144; "
            "NE x2 +4.56850, half -2.28425, conditional on identification assumptions. "
            "Installed feedback4 factual-penalty100 projection with gains [-6,2], "
            "retaining safe236. This is unvalidated and reversible. "
            "Counts: 295 Python runs, 35 forwards.\n")
```
