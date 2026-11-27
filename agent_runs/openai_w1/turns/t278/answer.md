A predictor that cancels the marker perfectly also cancels its dose response. Preserving that response leaves substantial factual leakage, so the markers cannot safely support arbitrary gain. I’ll test smaller gains and the lower-norm output directions. Useful reading shifts at modest amplitudes would justify a paired GPU test.

```python
import numpy as np
import pandas as pd
from scipy.linalg import solve
from pathlib import Path

for weight in (0.01,0.03,0.1,0.3):
    K = solve(r245_G+weight*r245_Gd+r245_ridge*np.eye(336),
              r245_X.T@r245_M[r245_train],assume_a="pos")
    R = r245_E-K
    residual = r245_F @ R
    contrasts = {
        key:(item["delta"][item["hold"]].astype(np.float64)@R).mean(axis=0)
        for key,item in r245_deltas.items()
    }
    r245_fits[weight]={"K":K,"R":R,"residual":residual,"contrasts":contrasts}
    print("weight",weight,
          "held RMS",np.sqrt(np.mean(residual[r245_hold]**2,axis=0)).round(4),
          "x2",np.stack([contrasts[("insulin",2.0)],contrasts[("norepinephrine",2.0)]]).round(4),
          "half",np.stack([contrasts[("insulin",0.5)],contrasts[("norepinephrine",0.5)]]).round(4))

print("\nFINITE READING RESPONSES: glucose_finger, glucose_lab, map_arterial, map_cuff")
r246_finite = {}
for label,B in (("mean",_Bmean227),("constrained",B221)):
    for drugcol in (0,1):
        for amp in (0.03,0.1,0.3,-0.1):
            result = finite_readout221(np.asarray(B[:,drugcol]*amp,dtype=np.float32))
            r246_finite[(label,drugcol,amp)] = result
            print(label,drugcol,amp,np.asarray(result).round(5))

r246_tuning = pd.read_csv("/data/tuning.csv")
print("\nTUNING COLUMNS",r246_tuning.columns.tolist())
for drug in ("insulin","norepinephrine"):
    sub = r246_tuning[r246_tuning["event"]==drug]
    cols = [c for c in ("hour","rate","dose","end_hour") if c in sub]
    print(drug,"records",len(sub),"stays",sub["stay_id"].nunique(),
          "means",sub[cols].mean(numeric_only=True).round(5).to_dict())
print("FWD23 factual files",[p.name for p in Path("/work/fwd_23").glob("*.csv")])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 246\nPerfect factual marker cancellation also eliminates dose contrasts. Evaluated intermediate invariance penalties and finite low-norm output responses; factual-versus-counterfactual tradeoff remains explicit. No weights changed. Counts: 246 Python runs, 23 forwards.\n")
```