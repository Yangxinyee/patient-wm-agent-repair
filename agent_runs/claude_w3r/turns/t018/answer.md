Both checks pass. The rows are exactly the recorded outcome readings at hours 13–24, and "reached" means a drug record started in [12, hour). On the same 100 tuning stays, my records-based linear impulse response gives an insulin ×2 readout of about −12.5 mg/dL and ×1/2 of about +6.2. The model says −0.53 and +0.09, so it underestimates insulin's effect roughly 20-fold.

A key design issue is the factual fit. Doses follow glucose bands, so the model's glucose dynamics already absorb the band-level insulin effect. Simply adding a dose-proportional term would double-count it and wreck the factual NLL. The edit has to act on the dose relative to what the policy would give (the deviation), which also keeps factual predictions intact. I will write this into the notes.

Next I will analyse the norepinephrine policy against MAP, in the same way as for insulin.

```python
with open('/work/notes.md','a') as f:
    f.write("""- FWD1 (original, tuning sel1 100 random stays): insulin_x2 -0.53, insulin_half +0.09, ne_x2 -1.33, ne_half -0.48 (n stays 55/21).
  Readout rows = recorded outcome readings (gf / map_arterial) at hours 13-24; reached = a drug record started in [12,hour). Verified.
  Records-based linear-IRF estimate on same rows: insulin x2 -12.5, x1/2 +6.25 -> model underestimates ~20x.
- KEY DESIGN ISSUE: dose ~ band(glucose); model's glucose dynamics absorb band-level insulin effect. Adding effect ∝ actual r would
  double count -> factual NLL disaster. Need edit acting on DEVIATION r - E[r|history] (z posterior = 'surprise' may encode it).
""")
ma = A['map_arterial']; mc = A['map_cuff']
mal, maa = ffill_last(ma); mcl, mca = ffill_last(mc)
rows=[]
for t in range(1,25):
    rows.append(pd.DataFrame({'t':t,'m':mal[:,t],'mage':maa[:,t],'mc':mcl[:,t],'r':ne_r[:,t],'rprev':ne_r[:,t-1],'stay':np.arange(n)}))
N_ = pd.concat(rows)
N_['m5']=np.floor(N_.m/5)*5
print(N_.groupby('m5').agg(n=('r','size'),p_on=('r',lambda x:(x>0).mean()),mean_r=('r','mean'),sd_r=('r','std')).round(3).to_string())
print('norepi given w/o arterial line:', ((N_.r>0)&N_.m.isna()).sum(), ' with:', ((N_.r>0)&N_.m.notna()).sum())
```