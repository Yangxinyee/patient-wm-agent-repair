No combination of ReLU and GELU in my head gets below 4 mg/dL, while the old helper reaches 1.86. So the activation is not the difference. It must be in the inputs.

The prime suspect is the "last value" register: which earlier reading counts as "last" (only the same code, or any glucose reading) and its age. Next I will compare, reading by reading, my last value and age with the `lastv`/`lasth` columns the old helper used. I will also feed `head_value2` my own GELU gap and z-prior to see whether its accuracy holds. If the last values disagree, that is the missing detail.

```python
F_ok = F_[F_.ok.astype(bool)]
key2i = {(int(r['stay_id']), round(float(r['hour']),3), r['event']): i for i, r in F_ok[['stay_id','hour','event']].iterrows()}
mism = 0; agem = 0; n_ = 0
for r_ in sub_r:
    kb, ev_, obs_, pm_, pv_, pa_ = r_
    i_ = key2i.get((int(S3[kb]), round(float(H3[kb]),3), ev_))
    if i_ is None: continue
    n_ += 1
    lv_, lh_ = F_.loc[i_, 'lastv'], F_.loc[i_, 'lasth']
    if not ((np.isnan(lv_) and np.isnan(pv_)) or abs(lv_ - pv_) < 1e-6): mism += 1
    if not np.isnan(pa_) and abs((H3[kb] - lh_)*3600 - pa_) > 1: agem += 1
print('rows', n_, 'lastv mismatch', mism, 'age mismatch', agem)
ex_ = [(r_[4], F_.loc[key2i[(int(S3[r_[0]]), round(float(H3[r_[0]]),3), r_[1]])], 'lastv']) for r_ in sub_r[:8] if (int(S3[r_[0]]), round(float(H3[r_[0]]),3), r_[1]) in key2i]
print('mine vs F lastv', [(round(a,1) if not np.isnan(a) else None, round(b,1) if not np.isnan(b) else None) for a,b in ex_])
```