Head 2 is the promising relay: its compensated factual output has a held-out RMS error of 0.040, versus 11.66 for the original attention output. The other heads are much harder to replace. I’ll install only this compensation and check GPU factual loss before adding a drug signal. A loss increase above 0.005 would reject the baseline.

```python
import numpy as np, torch, pandas as pd
from pathlib import Path

assert len(edited()) == 0, "Expected original weights before the relay baseline."
head207 = 2
fit207 = relay206["fits"][head207]
q207 = np205(load_original("core.prelude.0.attn.q.weight")).copy()
kv207 = np205(load_original("core.prelude.0.attn.kv.weight")).copy()
out207 = np205(load_original("core.prelude.0.attn.out.weight")).copy()

# Uniform attention in head 2; 46 compensated features, two unused features.
q207[96:144] = 0
kv207[192+96:192+144] = 0
kv207[192+96:192+142] = (
    fit207["left"][:,:46] * fit207["singular"][:46]).T.astype(np.float32)
out207[:,fit207["other"]] = fit207["coef"][:144].T.astype(np.float32)
out207[:,96:144] = 0
out207[:,96:142] = fit207["right"][:46].T.astype(np.float32)

relay207_base = {
    "core.prelude.0.attn.q.weight": q207,
    "core.prelude.0.attn.kv.weight": kv207,
    "core.prelude.0.attn.out.weight": out207
}
for name207, value207 in relay207_base.items():
    assert tuple(value207.shape) == tuple(SHAPES[name207])
    assert np.isfinite(value207).all()
    save_edit(name207, torch.from_numpy(value207.copy()))
print("Installed baseline:", edited())

print("\n=== MATCHED FACTUAL STAYS FOR NEXT FORWARD ===")
paths207 = [
    Path("/work/fwd_17/factual_original.csv"),
    Path("/work/fwd_15/factual_original.csv"),
    Path("/work/fwd_9/factual_original.csv")
]
for path207 in paths207:
    if path207.exists():
        df207 = pd.read_csv(path207)
        cols207 = [c for c in df207.columns if "stay" in c.lower()]
        print(path207.name, "columns", df207.columns.tolist())
        if cols207:
            relay207_stays = sorted(df207[cols207[0]].dropna().astype(int).unique().tolist())
            assert len(relay207_stays) <= 100
            print("stays =", relay207_stays)
            break
else:
    tuning207 = pd.read_csv("/data/tuning.csv", usecols=["stay_id"])
    relay207_stays = tuning207.stay_id.drop_duplicates().astype(int).tolist()[:100]
    print("first 100 tuning stays =", relay207_stays)

print("\n=== PRESERVED CARRIER STRUCTURE ===")
for drug207, spec207 in lc_branch.items():
    if isinstance(spec207,dict):
        print(drug207, {
            k: tuple(v.shape) if hasattr(v,"shape") else
               v if isinstance(v,(str,int,float,bool)) else type(v).__name__
            for k,v in spec207.items()
        })
    else:
        print(drug207, type(spec207).__name__)
    if drug207 in lc_q:
        a207 = np205(lc_q[drug207])
        print("factual carrier quantiles", np.quantile(a207,[0,.01,.5,.99,1]))

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 207\nInstalled compensated uniform prelude attention head 2, with 46 factual features and two unused carrier features. Held-out local attention RMSE .03996 versus original RMS 11.655. No drug carrier installed yet; GPU factual validation is required. Counts: 207 Python runs, 18 forwards.\n")
```