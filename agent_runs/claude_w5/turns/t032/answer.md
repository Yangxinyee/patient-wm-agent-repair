The tensor metadata contains only shapes, so the exact computation is known only from ARCHITECTURE.md.

That settles the design constraint. Doses reach glucose and MAP predictions only through the recurrent state, so any edit must either:
- read dose information already present in the state and shift the value-head outputs, or
- make the core encode dose effects.

The edit must also add only the dose's deviation from what the policy would have given, Σ κ_j (u − ū(readings)). The original model already absorbs the average dose effect through its dependence on readings. Adding Σ κ_j u in full would bias factual predictions by several mg/dL and break the 0.005-nat limit.

Next, cheap work in Python before spending forward passes:
1. Compute the g-estimation-based target readouts over all 30,000 training stays, using the evaluation's reached-row rule. This gives the expected truth on unseen stays and how it varies across 100-stay samples.
2. Bootstrap the γ profiles over stays, to know how precise the targets are.

```python
def target_readout(Pdf, gam, drug, outcome, factor, stays=None):
    d = Pdf if stays is None else Pdf[Pdf.stay_id.isin(stays)]
    Rg = d.pivot(index='stay_id', columns='h', values=drug).reindex(columns=range(-2,26))
    Yg = d.pivot(index='stay_id', columns='h', values=outcome).reindex(columns=range(-2,26))
    U = np.nan_to_num(Rg.values); has = ~np.isnan(Rg.values); Y = ~np.isnan(Yg.values)
    cols = list(Rg.columns)
    i12 = cols.index(12)
    vals = []; per_stay = []
    for i in range(U.shape[0]):
        drug_idx = [c for c in range(i12, cols.index(25)) if has[i, c]]
        if not drug_idx: continue
        first = cols[drug_idx[0]]
        sv = []
        for T in range(13, 25):
            if T <= first or not Y[i, cols.index(T)]: continue
            sft = sum(gam[T - cols[c]]*U[i, c] for c in range(i12, cols.index(T)) if (T-cols[c]) < len(gam))
            sv.append((factor-1)*sft)
        vals += sv
        if sv: per_stay.append(np.mean(sv))
    return np.mean(vals), len(vals), len(per_stay), np.array(vals)
for nm, drug, outc, gm in [('ins','insulin','glucose_finger',gam_all), ('ne','norepinephrine','map_arterial',gam_ne)]:
    m2, n2, ns, v2 = target_readout(P, gm, drug, outc, 2.0)
    print(nm, "x2 target over train:", round(m2,3), "rows", n2, "stays", ns, " half:", round(-0.5*m2,3))
    # variability across random 100-stay tuning-like samples
    sids = P.stay_id.unique(); rng = np.random.default_rng(0); sims=[]
    for rep in range(20):
        ss = rng.choice(sids, 400, replace=False)
        sims.append(target_readout(P, gm, drug, outc, 2.0, stays=set(ss))[0])
    print("   sd across 400-stay samples:", round(np.std(sims),3))
```