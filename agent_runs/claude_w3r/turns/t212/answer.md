Run 199 fixes the compensation forms.

**Glucose (residual sd 14.6 mg/dL).** The original model misses the insulin deviation effects:

| Term | Coefficient (se) |
|---|---|
| lag 1 | −1.43 (0.55) |
| lag 2 | −1.21 (0.58) |
| lag 3 | −0.40 (0.60) |

This fits the record-based −2.0/−0.95/−0.26 given partial capture. I'll use the deviation form −2.0·(r_s − m) − 0.95·(r_{s−1} − m). The band compensation goes through the last glucose bin, (2.0+0.95)·m(g_last), since the previous reading isn't in the register.

**MAP (residual sd 4.5 mmHg).** The model misses about +3.1 per unit of lag-1 norepinephrine. After adding 3.45·r1 + 0.72·r2, the residual needs −0.93 when norepinephrine is on (plus a general −1.4 bias in these sick stays, which I'll leave alone). So the norepinephrine channel is 3.45·ne_s + 0.72·ne_e − 0.93·[on].

My injections are fast-decaying, so they only reach a reading if the action bundle at s:05 is the bundle just before it. Next I'll check bundle timing: the bundle times within the hour, and for each glucose_finger and map_arterial reading the kind and offset of the preceding bundle. If most readings directly follow an s:05 action bundle, the design stands. If cuff or lab bundles often fall in between, the channel would be lost for those readings and needs more persistence.

```python
PD['tm']=np.round((PD.t%1)*60).astype(int)
print('bundle minute-of-hour counts', PD.tm.value_counts().head(10).to_dict())
# for each gf / map_arterial reading bundle: what is the previous bundle?
prev_info=[]
for s in stays_S[:100]:
    Bs_,_,_=stay_data(s)
    for k in range(1,len(Bs_)):
        codes_=[a['code'] for a in Bs_[k][1]]
        for cc_,nm_ in [(6,'gf'),(7,'map_art')]:
            if cc_ in codes_:
                pa=Bs_[k-1][1]; pk='action' if any(a['rate'] is not None and a['kind']==0 for a in pa) else ('endonly' if any(a['rate'] is not None for a in pa) else 'reading/other')
                prev_info.append((nm_,pk,round((Bs_[k][0]-Bs_[k-1][0])*60)))
pi_=pd.DataFrame(prev_info,columns=['ev','prev','gapmin'])
print(pi_.groupby(['ev','prev']).size().unstack(fill_value=0))
print(pi_.groupby('ev').gapmin.value_counts().groupby(level=0).head(6))
```