This repeats the frontier result; the matched target calculation has not returned yet. The baseline’s −10.83/+3.36 shifts must be compared with stay-weighted targets, not pooled-hour targets. I’ll calculate that comparison now. If doubling is already near its target, preserving it will be more valuable than blindly increasing edit strength.

```python
import numpy as np, pandas as pd, inspect, re
from pathlib import Path

tuning214 = pd.read_csv("/data/tuning.csv")
schedule214 = {}
for drug in ("insulin", "norepinephrine"):
    d = tuning214[tuning214.event.eq(drug)].copy()
    d["dose_hour"] = np.floor(d["hour"].to_numpy()+1e-6).astype(int)
    schedule214[drug] = {
        int(sid): (g["dose_hour"].to_numpy(),g["rate"].to_numpy())
        for sid,g in d.groupby("stay_id")
    }

def targets214(path, insulin_beta=-2.2778, insulin_phi=.85743,
               ne_beta=3.1670730232, ne_phi=.704):
    d = pd.read_csv(path)
    d = d[d["reached"].astype(str).str.lower().isin(["true","1"])].copy()
    result = {}
    for edit,g in d.groupby("edit"):
        drug = "insulin" if edit.startswith("insulin") else "norepinephrine"
        beta,phi = (insulin_beta,insulin_phi) if drug=="insulin" else (ne_beta,ne_phi)
        multiplier = 2.0 if edit.endswith("x2") else .5
        pred = []
        for r in g.itertuples():
            hours,rates = schedule214[drug].get(int(r.stay_id),
                         (np.array([],dtype=int),np.array([],dtype=float)))
            lag = int(r.hour)-hours-1
            active = (hours>=12)&(lag>=0)
            pred.append(beta*(multiplier-1)*np.sum(rates[active]*phi**lag[active]))
        g["target"] = pred
        result[edit] = {
            "stays": int(g.stay_id.nunique()), "hours": len(g),
            "target_stay_weighted": round(float(g.groupby("stay_id")["target"].mean().mean()),3),
            "target_hour_pooled": round(float(g["target"].mean()),3),
            "model_stay_weighted": round(float(g.groupby("stay_id")["shift"].mean().mean()),3)
        }
    return result

print("=== MATCHED RECORD-DERIVED TARGETS ===")
for n in (1,10,15,18):
    model = "original" if n==1 else "edited"
    print("forward",n,targets214(f"/work/fwd_{n}/dose_edit_{model}.csv"))

print("\n=== RELEVANT HELPER SIGNATURES ===")
for name,v in list(globals().items()):
    if callable(v) and getattr(v,"__module__","")=="__main__" and re.search(
        r"(static|register|meta|decoder|states|decode|build|head_calc|schedule_readout)",name):
        print(name,inspect.signature(v))

print("\n=== PARAMETER SENSITIVITY, FIRST SAMPLE ===")
for ib,ip,nb,ph in [(-2.12,.83,2.8,.65),(-2.2778,.85743,3.167,.704),
                    (-2.44,.88,3.5,.75)]:
    a = targets214("/work/fwd_1/dose_edit_original.csv",ib,ip,nb,ph)
    print((ib,ip,nb,ph),{k:v["target_stay_weighted"] for k,v in a.items()})
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 214: align causal target weighting with evaluation\n"
            "Baseline second-sample stay-weighted shifts: insulin -10.831/+3.357; NE +.098/-.955. "
            "Computing impulse-model targets on identical reached rows, averaging within stays "
            "then equally across stays. Parameter sensitivity is not ground truth. "
            "No weights changed. Python runs: 214; forwards: 23.\n")
```
