The sliding scale is a clean step function of the reading taken that hour, finger or lab, with the same cut-points for both:

| Reading (mg/dL) | Base rate (u/h) |
|---|---|
| below 180 | 1 |
| 180–220 | 2 |
| 220–260 | 3 |
| 260–300 | 4 |
| 300 and above | 6 |

The administered rate is the base rate times noise, rounded to 0.5.

Next I'll test whether that noise, log(rate/base), is random. I'll check it for:
- persistence within a stay, which would suggest a clinician or patient factor;
- correlation with recent glucose trend, MAP, norepinephrine or hour.

If it is unrelated to everything except possibly itself, the noise-based regression gives a valid causal estimate. If it tracks hidden patient state, I'll need to lean on the discontinuities at 180/220/260/300.

```python
def base_ins(g):
    g = np.asarray(g, float)
    b = np.where(g<180,1.0,np.where(g<220,2.0,np.where(g<260,3.0,np.where(g<300,4.0,6.0))))
    return b
P['g_now'] = P.glucose_finger.fillna(P.glucose_lab)
P['src_g'] = np.where(P.glucose_finger.notna(),'f',np.where(P.glucose_lab.notna(),'l','none'))
P['ins_base'] = base_ins(P.g_now)
P['ins_noise'] = np.log(P.insulin/P.ins_base)
g = P.groupby('stay_id')
P['ins_noise_prev'] = g.ins_noise.shift(1)
P['g_prev'] = g.g_now.shift(1)
P['g_now_lag2'] = g.g_now.shift(2)
on = P[(P.ins_on==1)&P.g_now.notna()].copy()
print("noise by src:", on.groupby('src_g').ins_noise.agg(['mean','std','size']).to_string())
print(on.groupby('ins_base').ins_noise.agg(['mean','std','size']).to_string())
print(np.exp(on.ins_noise).round(3).value_counts().sort_index().head(30).to_string())
cc = on[['ins_noise','ins_noise_prev','g_now','g_prev','g_now_lag2','map_arterial','map_cuff','norepinephrine','h']].corr()
print(cc['ins_noise'].to_string())
```