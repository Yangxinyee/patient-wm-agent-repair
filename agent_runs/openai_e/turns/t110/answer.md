Factor-one re-encoding now reproduces the drug posterior exactly. The corrected nonlinear detector still has a weak halving response. More importantly, the earlier factual action encodings were also inconsistent. I’ll check whether correcting those inputs improves the simpler recurrent detector before attempting more complex surgery. Lower factual activation on the corrected features would support that explanation for the failed NE trial.

```python
import numpy as np, torch
from scipy.linalg import null_space
from pathlib import Path

E_fact95 = core_norm(
    torch.as_tensor(raw_fact94, dtype=torch.float32),
    "core.prelude_norm").numpy()
E_edit95 = {}
for key in Y81:
    action = edited_action46(*key)
    E_edit95[key] = prelude46(
        hflow46[idx93], cf_E, action,
        posterior94(cf_C, cf_E, action)).numpy()

print("CORRECTED VERSUS LEGACY PRELUDE")
print("All-bundle RMS", float(np.sqrt(np.mean(
    (E_fact95-prelude_all46.numpy())**2))))
print("Drug-bundle RMS", float(np.sqrt(np.mean(
    (E_fact95[idx93]-prelude_cf46.numpy())**2))))

old_feature95 = np.asarray(ne_feature82, dtype=float)
flat_E95 = E_fact95.reshape(-1,192).astype(float)
old_factual95 = flat_E95[test_fact_mask93] @ old_feature95
print("Old detector corrected factual RMS",
      float(np.sqrt(np.mean(old_factual95**2))))
for key in Y81:
    delta = (E_edit95[key]-E_fact95[idx93]).reshape(-1,192)
    p = delta[test_cf_mask93] @ old_feature95
    y = np.asarray(Y81[key])[test_cf_mask93]
    if key[0] == "norepinephrine":
        print("Old detector corrected slope",key,float(p@y/(y@y)))
    else:
        print("Old detector corrected cross RMS",key,
              float(np.sqrt(np.mean(p*p))))

scale95 = np.maximum(flat_E95[fit_fact_mask93].std(axis=0),0.02)
with np.load("/work/insulin_candidate58.npz") as z:
    initial95 = z["initial_basis"].astype(float)
null95 = null_space(initial95/scale95, rcond=1e-6)
XF95 = (flat_E95/scale95) @ null95
GF95 = XF95[fit_fact_mask93].T @ XF95[fit_fact_mask93] / fit_fact_mask93.sum()
D95,G95,R95 = {},{},{}
for key in Y81:
    delta = (E_edit95[key]-E_fact95[idx93]).reshape(-1,192)
    D95[key] = (delta/scale95) @ null95
    d = D95[key][fit_cf_mask93]
    y = np.asarray(Y81[key],float)[fit_cf_mask93]
    G95[key] = d.T@d/len(d)
    R95[key] = d.T@y/len(d)

fits95,metrics95 = {},[]
for cross in [1.,10.,100.]:
    for half_weight in [1.,4.,10.]:
        for anchor in [30.,300.,3000.]:
            weights = {
                ("norepinephrine",2.):1.,
                ("norepinephrine",.5):half_weight,
                ("insulin",2.):cross,("insulin",.5):cross}
            gram = anchor*GF95 + .001*np.eye(null95.shape[1])
            rhs = np.zeros(null95.shape[1])
            for key,w in weights.items():
                gram += w*G95[key]
                if key[0]=="norepinephrine":
                    rhs += w*R95[key]
            theta = np.linalg.solve(gram,rhs)
            feature = (null95@theta)/scale95
            fits95[(cross,half_weight,anchor)] = feature
            fp = flat_E95[test_fact_mask93]@feature
            m = {"choice":(cross,half_weight,anchor),
                 "factual_rms":float(np.sqrt(np.mean(fp*fp)))}
            for key in D95:
                p = D95[key][test_cf_mask93]@theta
                if key[0]=="norepinephrine":
                    y = np.asarray(Y81[key])[test_cf_mask93]
                    m["s"+str(key[1])] = float(p@y/(y@y))
                else:
                    m["cross"+str(key[1])] = float(np.sqrt(np.mean(p*p)))
            s=m["s2.0"]
            m["unit_factual"]=m["factual_rms"]/max(abs(s),1e-9)
            m["unit_cross"]=max(m["cross2.0"],m["cross0.5"])/max(abs(s),1e-9)
            m["half_ratio"]=m["s0.5"]/max(abs(s),1e-9)
            m["score"]=(m["unit_factual"]+.04*m["unit_cross"]+
                        .12*abs(m["half_ratio"]-1)+(10 if s<=0 else 0))
            metrics95.append(m)

print("\nCORRECTED RECURRENT DETECTORS")
for m in sorted(metrics95,key=lambda x:x["score"])[:10]:
    print(m)

print("\nSTATE COHORT")
print("IDs",cf_frame["stay_id"].min(),cf_frame["stay_id"].max(),
      "number",cf_frame["stay_id"].nunique())
print("Factual files")
for path in sorted(Path("/work").glob("fwd_*/factual_original.csv")):
    import pandas as pd
    f = pd.read_csv(path)
    print(str(path),f.shape,list(f.columns),
          (f["stay_id"].min(),f["stay_id"].max()) if "stay_id" in f else "")

with open("/work/notes.md","a") as f:
    f.write(
        "\n## Run 95: corrected recurrent feature audit\n"
        "Factor-one drug posterior is exact. Rebuilt normalized prelude features "
        "with consistent drug observation/action encodings and true factual "
        "posterior means. Audited old NE detector leakage and refitted constrained "
        "192-feature detectors before any new weight edit. "
        "Python runs: 95; forwards: 12.\n")
```