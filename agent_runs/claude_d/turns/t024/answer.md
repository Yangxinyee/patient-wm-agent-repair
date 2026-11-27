The g-estimated insulin impulse response is clean. With all other doses held fixed, 1 u/h at hour s lowers glucose_finger by:

| lag (h) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mg/dL | −1.99 | −2.39 | −2.11 | −1.71 | −1.21 | −0.92 | −0.64 | −0.47 | −0.20 | −0.13 | −0.08 | −0.01 |

Standard errors are 0.05–0.16. So the effect peaks at 2 h and fades with a half-life of about 2–3 h. A sustained extra 1 u/h gives a steady state of about −11.9 mg/dL.

Under a linear, additive response, the true shift at a reached hour t is Σ_k β_k × Δins_{t−k}, counting doses from hour 12 on. Δins is the recorded rate for x2 and minus half of it for x1/2.

Next I will compute this implied true readout on the same 100 tuning stays and stay-hours the tool reported, and compare it with the model's −1.73 and +0.91. I expect roughly −8 to −12 for x2, about 5–7 times the model's answer.

```python
beta_ins = beta.copy()
json.dump({'beta_ins': beta_ins.tolist()}, open('/work/beta_ins.json','w'))
stU = PUu.index.get_level_values('stay_id').unique()
def arrU(P_, col): return P_[col].values.reshape(len(P_.index.get_level_values('stay_id').unique()), H)
INS_U = arrU(PUu,'ins0'); GF_U = arrU(PUu,'gf'); NOR_U = arrU(PUu,'nor0'); MA_U = arrU(PUu,'ma')
sid2i = {s:i for i,s in enumerate(stU)}
def true_shift_ins(sid, hour, factor, beta_):
    i = sid2i[sid]; t = hour+1  # index
    dI = INS_U[i]*(factor-1); dI[:13] = 0   # doses from hour 12 (index 13) on
    return sum(beta_[k]*dI[t-k] for k in range(1, min(K, t-13)+1) if t-k>=13)
de['true'] = np.nan
for e_, f_ in [('insulin_x2',2),('insulin_half',0.5)]:
    m_ = de.edit==e_
    de.loc[m_,'true'] = [true_shift_ins(s,h,f_,beta_ins) for s,h in zip(de.stay_id[m_], de.hour[m_])]
r = de[de.reached]
print(r.groupby('edit')[['shift','true']].mean())
r['since'] = np.nan
print(r[r.edit=='insulin_x2'].groupby('hour')[['shift','true']].mean().round(2).T.to_string())
```