The core layout settles how dose memory could work:
- `c_out` is a plain 192×192 matrix.
- `coda`, the two recurrent blocks and the prelude are all pre-norm residual transformer layers.
- So a dose memory lasting about 12 hours would need a direction in h that every step carries forward almost unchanged.
- Building that needs an exact replica of the step, or careful finite-difference measurements through the tool.

Before investing in that, I'll put a number on the factual-fit cost of the simpler alternative. I'll compare adding a correction on the raw dose, Σ c_j·d_{t−j}, against one on the residual, Σ c_j·ε_{t−j}. The c_j come from the error regression (−1.36, −1.14, −0.62 for insulin). The change in mean squared error divided by twice the MSE approximates the change in NLL per reading.
- **If the raw-dose version costs well over 0.005 nats per reading** (pooled over all 4,989 readings), plain amplification is ruled out and the residual design is needed.

```python
def corr_effect(ev, drug, c):
    m=fc[(fc.event==ev)&(fc.hour>=1)]
    si=m.stay_id.map(sidx_tu).values; t=np.round(m.hour.values).astype(int)
    e=(m.observed-m.predicted_mean).values
    raw=np.zeros(len(e)); res=np.zeros(len(e))
    for L,cl in enumerate(c,1):
        dd,pp=dosemat(drug,si,t,L); raw+=cl*dd; res+=cl*(dd-pp)
    mse=np.mean(e**2)
    out=[]
    for nm,cc in [('raw',raw),('res',res)]:
        d_mse=np.mean((e-cc)**2)-mse
        out.append('%s: dMSE %.2f  approx dNLL/reading %.4f'%(nm,d_mse,d_mse/(2*mse)))
    print(ev, 'n',len(e),'mse %.1f'%mse,' | '.join(out))
    return len(e)
tot=4989
for ev,dr,c in [('glucose_finger','insulin',[-1.36,-1.14,-0.62]),('glucose_lab','insulin',[-1.36,-1.14,-0.62]),
                ('map_arterial','norepinephrine',[3.2,2.0,0.5]),('map_cuff','norepinephrine',[3.2,2.0,0.5])]:
    corr_effect(ev,dr,c)
```