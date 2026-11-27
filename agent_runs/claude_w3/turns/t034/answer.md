This confirms the diagnosis.
- In factual mode the model's predicted next glucose does not respond to the insulin jitter: the coefficient is +0.45 ± 0.32, so if anything slightly upward. The observed next reading responds with −1.57 ± 0.67, in line with the −2 estimated on train.
- The model's next prediction is roughly 0.49 × last reading + 0.49 × last prediction. That is an exponential smoother with gain about 0.5 and weights summing to about 1, so it carries any level shift forward almost indefinitely.

This matters for the edit design. In a rollout, a tilt added at the value head shifts the drawn reading. The smoother then carries about half of that tilt into its base belief, permanently. So the per-lag tilt has to follow the increments of the true response, adjusted for that integration: strongly negative in the first 2–3 hours after a dose, slightly positive later to produce the washout. The tilt filter then sums to about zero, so in factual data it is near zero under steady dosing. That should keep the factual NLL penalty small even without explicit centring.

Before modelling the state, I want two more facts about the truth:
1. Cross effects: insulin jitter on MAP, and norepinephrine innovation on glucose. The model's feedback could pick these up.
2. Whether the insulin effect per unit depends on the glucose level, which would make ×2 and ×½ asymmetric.

I will run the insulin IV on map_arterial, the norepinephrine IV on lab glucose, and the insulin one-hour finger effect by glucose band with pol × dose interactions. Cross effects near zero and a level-proportional insulin effect would confirm my current picture.

```python
# cross effects via IV
MAPA_s = MAPA  # norepi arrays built on x (same stay order as pn0?)
same_order = np.array_equal(x.stay_id.values.reshape(SS,25)[:,0], pn0.stay_id.values.reshape(S,25)[:,0])
print("same stay order:", same_order)
def iv_generic(L, Y, Dm_all, Zm_all, origin_mask, ctrl):
    rows=[]
    for t in range(0, 25-L):
        ok = origin_mask[:,t] & ~np.isnan(Y[:,t+L])
        if ok.sum()==0: continue
        rows.append(np.column_stack([np.full(ok.sum(), t), ctrl[ok,t], Y[ok,t+L], Dm_all[ok,t:t+L], Zm_all[ok,t:t+L]]))
    Am = np.vstack(rows)
    grp = pd.Series(list(zip(Am[:,0].astype(int), np.nan_to_num(Am[:,1]).astype(int)))).factorize()[0]
    df = pd.DataFrame(Am[:,2:]); dm = (df - df.groupby(grp).transform('mean')).values
    yv = dm[:,0]; Dv = dm[:,1:1+L]; Zv = dm[:,1+L:]
    PZ = Zv @ np.linalg.lstsq(Zv, Dv, rcond=None)[0]
    b = np.linalg.lstsq(PZ, yv, rcond=None)[0]; res = yv - Dv@b; Vv = np.linalg.inv(PZ.T@PZ)*res.var()
    return b, np.sqrt(np.diag(Vv)), len(yv)
polbin = np.where(np.isnan(POL), -1, POL//2)
for L in [1,3,6]:
    b,se,n = iv_generic(L, MAPA, INS, Z, POL>150, polbin)
    print(f"insulin -> map_arterial L={L}: lag-L coef {b[0]:.3f} ({se[0]:.3f}); lag1 coef {b[-1]:.3f} ({se[-1]:.3f}); sum {b.sum():.2f} n={n}")
for L in [1,3,6]:
    b,se,n = iv_generic(L, GL, NR, ZR, NR>0, KEYC)
    print(f"norepi -> glucose_lab L={L}: lag-L coef {b[0]:.3f} ({se[0]:.3f}); lag1 {b[-1]:.3f} ({se[-1]:.3f}); sum {b.sum():.2f} n={n}")
# insulin one-step effect by band with interaction (finger outcome, train)
s['zj'] = s.insulin - mtab_full.values[np.clip(np.round(s.glucose_finger).astype(int),0,599)]
for lo,hi in [(150,180),(180,220),(220,260),(260,400)]:
    sb = s[(s.glucose_finger>lo)&(s.glucose_finger<=hi)]
    X = np.column_stack([np.ones(len(sb)), sb.glucose_finger, sb.zj])
    bb = np.linalg.lstsq(X, sb.dG.values, rcond=None)[0]; rr = sb.dG.values-X@bb; se_=np.sqrt(np.diag(np.linalg.inv(X.T@X))*rr.var())
    print(f"band {lo}-{hi}: one-step coef per unit {bb[2]:.3f} ({se_[2]:.3f}), mean G {sb.glucose_finger.mean():.0f}, per-unit/G*100 {100*bb[2]/sb.glucose_finger.mean():.3f}")
```