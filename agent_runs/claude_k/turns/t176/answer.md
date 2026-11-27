The embedding has more categorical tables than I remember using: action (2 rows), rate_unit (4), route (2), status (2), unit (6), plus text and kq.

The true drug atom may include rows my rebuild leaves out, such as an `action` flag, a `rate_unit` row, a route or status row, or a dose-unit row in `unit`. Any of those would shift u in a way the sensitive decoder amplifies.

First I need to know what my `drug_atoms` actually builds. I'll decompose its insulin atom (rate 1.5, z = 0) into embedding rows:
- subtract the code, kind, ekind and numeric terms;
- match the residual against every row of every table;
- check which bin rows fit.

```python
T={n_:np.asarray(load_original(n_)) for n_ in NAMES if n_.startswith('embed.')}
num_W=T['embed.numeric.weight']; num_b=T['embed.numeric.bias']
zr_=(1.5-1.5)/1.0
known=T['embed.code.weight'][8]+T['embed.kind.weight'][0]+num_b+num_W@np.array([0,0,zr_,1,zr_,1.0])
best_=[]
for ek in range(5):
    r_=V0-known-T['embed.ekind.weight'][ek]
    # try bins
    for bd in range(17):
        for brt in [bd]:
            rr_=r_-T['embed.bins.dose.weight'][bd]-T['embed.bins.rate.weight'][brt]
            best_.append((np.linalg.norm(rr_),ek,bd,brt))
best_.sort(); print('best (resid, ekind, dosebin, ratebin):', [tuple(np.round(b,3) if isinstance(b,float) else b for b in x) for x in best_[:4]])
ek,bd,brt=best_[0][1:]
rr_=V0-known-T['embed.ekind.weight'][ek]-T['embed.bins.dose.weight'][bd]-T['embed.bins.rate.weight'][brt]
for tn in ['embed.categorical.action.weight','embed.categorical.rate_unit.weight','embed.categorical.route.weight','embed.categorical.status.weight','embed.categorical.unit.weight','embed.categorical.discharged_to.weight','embed.text.weight','embed.kq.weight']:
    Wt=T[tn]; print(tn, 'resid norm if subtract row r:', [round(float(np.linalg.norm(rr_-Wt[r])),3) for r in range(Wt.shape[0])], '| none', round(float(np.linalg.norm(rr_)),3))
```