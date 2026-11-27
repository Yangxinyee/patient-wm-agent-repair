Insulin has a real but modest effect. Each extra 1 u/h during hour t lowers the next-hour glucose_finger by about 1.98 mg/dL (SE 0.05). The placebo is clean (−0.02 ± 0.05), so the within-band deviation behaves as random.

Next I will measure how the effect carries over to later hours. I will regress g_{t+s}, s = 1..8, on the deviation x_t with the same controls. This reduced form includes some feedback through later dosing, but it shows the effect's persistence: whether it accumulates and decays slowly, or is transient. That determines how a sustained ×2 dose from hour 12 adds up over hours 13–24.

```python
with open('/work/notes.md','a') as f:
    f.write("""- DATA: readings hourly (int hours -1..24), infusions start t+5min, 1h long. Weight fixed 80kg -> dose∝rate. All stays 25.17h.
- INSULIN POLICY: banded sliding scale on last glucose_finger: <150 off; 150-180 base1; 180-220 base2; 220-260 base3; 260-300 base4; >=300 ~6.
  Within-band rate = multiplicative noise (log-sd ~0.35), i.i.d. across hours (lag1 corr 0.008) -> as-if random.
- RESULT: one-hour effect of +1 u/h insulin deviation on next-hour glucose_finger: -1.98 (se .05) mg/dL; placebo on g_{t-1}: -0.02 (se .05).
""")
for s in range(1,9):
    rows=[]
    for t in range(1,T-s):
        rows.append(pd.DataFrame({'g':gf[:,t],'y':gf[:,t+s],'r':ins_r[:,t],'stay':np.arange(n)}))
    F=pd.concat(rows).dropna(subset=['g']); F['g5']=np.floor(F.g/5)*5
    F['x']=F.r-F.groupby('g5').r.transform('mean'); F['g2']=(F.g-160)**2/100
    b,s_,nn=ols(F,'y',['x','g','g2'])
    print('s=%d coef %.3f se %.3f n %d'%(s,b.x,s_.x,nn))
```