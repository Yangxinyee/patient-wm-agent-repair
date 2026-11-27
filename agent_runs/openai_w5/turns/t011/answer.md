The saved analysis controls for recorded history, but it has not yet checked whether effects differ by sensor or dose range. The multi-hour responses also do not fit a simple one-step decay perfectly. I’ll test those points before fixing rollout targets. Consistent estimates across sensors and dose ranges would support a shared, approximately linear drug effect; disagreement would call for a narrower or nonlinear repair.

```python
import inspect
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

print("=== NORMALISATION AND HEAD DETAILS ===")
for entry in VOCAB["numeric"]:
    print(entry)
print("additional names:", [n for n in NAMES if n.startswith(("core.gap","gap","embed.kq"))])
print("OLS signature:", inspect.signature(clustered_ols))
print("mask hours:", np.unique(hours[fitmask]), np.unique(hours[testmask]))
print("treatment residual shapes:", {k: np.shape(v) for k,v in residuals.items()})
print("last_numeric column norms:", np.round(np.linalg.norm(np.asarray(W0["last_numeric.weight"]),axis=0),3))
print("W0 types:", type(W0["last_numeric.weight"]), type(load("last_numeric.weight")))

test_positions = np.flatnonzero(testmask)
test_index = panel.index[testmask]
rt = np.column_stack([residuals["insulin"], residuals["norepinephrine"]])
print("\n=== TREATMENT SUPPORT ===")
for drug in ["insulin","norepinephrine"]:
    rates = panel.loc[hours >= 0, drug]
    print(drug, "positive fraction", round(float((rates>0).mean()),3),
          "positive quantiles", np.round(rates[rates>0].quantile([0,.1,.5,.9,.99,1]).values,3))
    r = residuals[drug]
    print("held-out innovation SD", round(float(np.nanstd(r)),4),
          "innovation mean", round(float(np.nanmean(r)),4))

sensor_checks = {}
with threadpool_limits(limits=8):
    for event in ["glucose_finger","glucose_lab","map_arterial","map_cuff"]:
        ynext = panel[event].groupby(level=0).shift(-1).to_numpy()
        fm = fitmask & np.isfinite(ynext)
        nuisance = HistGradientBoostingRegressor(max_iter=140, max_leaf_nodes=25,
                    l2_regularization=15, learning_rate=.07, random_state=17)
        nuisance.fit(X.loc[fm], ynext[fm])
        resid = ynext[testmask] - nuisance.predict(X.loc[testmask])
        drug = "insulin" if event.startswith("glucose") else "norepinephrine"
        jdrug = 0 if drug == "insulin" else 1
        valid = np.isfinite(resid) & np.isfinite(rt).all(axis=1)
        bcheck, scheck, ncheck = clustered_ols(rt[valid], resid[valid], testsid[valid])
        sensor_checks[event] = (bcheck,scheck,ncheck)
        print("\n",event, "next-hour effects [insulin, NE]:",
              np.round(bcheck,4), "cluster SE", np.round(scheck,4), "n", ncheck)
        rawrate = panel[drug].to_numpy()[testmask]
        for lo,hi in ([(0,2),(2,5),(5,50)] if drug=="insulin" else [(0,.15),(.15,.4),(.4,2)]):
            vm = valid & (rawrate>=lo) & (rawrate<hi)
            if vm.sum()>500:
                bb,ss,nn = clustered_ols(rt[vm],resid[vm],testsid[vm])
                print(" rate",lo,hi,"effect",round(float(bb[jdrug]),4),
                      "SE",round(float(ss[jdrug]),4),"n",nn)

print("\n=== ALTERNATIVE DYNAMIC RESPONSE FITS ===")
dynamic_alternatives = {}
for label,drugletter in [("G","I"),("M","N")]:
    yr = np.array([0.] + [lp[(label,h)][0][0 if label=="G" else 1] for h in range(1,9)])
    ar = np.array([lp[(drugletter,h)][0][0 if label=="G" else 1] for h in range(8)])
    print(label,"outcome response",np.round(yr,3),"dose response",np.round(ar,3))
    design2 = np.column_stack([yr[:-1],ar,np.r_[0.,ar[:-1]]])
    coeff2 = np.linalg.lstsq(design2,yr[1:],rcond=None)[0]
    kernel = np.empty(8)
    for h in range(8):
        kernel[h] = yr[h+1] - sum(kernel[h-l]*ar[l] for l in range(1,h+1))
    dynamic_alternatives[label] = {"arx":coeff2,"kernel":kernel}
    print(" [persistence, current dose, lagged dose]:",np.round(coeff2,4))
    print(" deconvolved impulse kernel:",np.round(kernel,4))

with open("/work/notes.md","a") as f:
    f.write("\n## Restart inspection and validation (runs 6–8)\n"
            "- Re-read architecture and recovered persistent analysis objects; no edits.\n"
            "- Exploring action-pathway versus value-head edits, with explicit factual-fit protection.\n"
            "- Sensor-specific and dose-stratified causal checks, plus distributed-lag alternatives, printed in run 8.\n")
```
