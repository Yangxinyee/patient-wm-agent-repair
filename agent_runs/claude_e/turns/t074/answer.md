The best variant so far (measurement type index 1, register layout A) has correlation 0.992 but MAE 3.5 mg/dL, so something is still mis-specified. Next I will vary the remaining guesses one at a time:

- age units (seconds vs hours)
- the gap input scaling
- z handling (prior mean only vs draws, and the order of the mean and log-sd halves)
- strict vs non-strict bin edges

I will report MAE per event so I can see which code drives the error. A variant with MAE near zero across all four codes would pin the head down.

```python
def mae_by_event(pm):
    e = np.abs(pm - R.pm.values)
    return {ev: round(float(e[R.event.values==ev].mean()),3) for ev in ['glucose_finger','glucose_lab','map_arterial','map_cuff']}
base = dict(type_idx=1, feat_mode='A')
pm,_ = predict(R, **base); print('base', mae_by_event(pm))
pm,_ = predict(R, **base, age_div=10.0*1.0, nz=1, seed=1); print('nz=1', mae_by_event(pm))
# age in hours
Rh = R.copy(); Rh['age'] = R.age/3600.0
pm,_ = predict(Rh, **base); print('age hours', mae_by_event(pm))
for gd in [1.0, 5.0, 20.0]:
    pm,_ = predict(R, **base, gap_div=gd); print('gap_div', gd, mae_by_event(pm))
# z prior: mean only (no noise)
def predict_zmean(R, swap=False, **kw):
    global z_draws
    old = z_draws
    def zd(ctx, nz, gen, act='gelu'):
        o = mlp2(ctx, 'z_prior', act); mu, lsd = (o[:, 32:], o[:, :32]) if swap else (o[:, :32], o[:, 32:])
        return mu[None] + (0 if kw.get('nonoise') else torch.exp(lsd)[None]*torch.randn((nz,)+mu.shape, generator=gen, dtype=torch.float64))
    z_draws = zd
    try:
        kk = {k:v for k,v in kw.items() if k!='nonoise'}
        return predict(R, **base, **kk)
    finally:
        z_draws = old
pm,_ = predict_zmean(R, nonoise=True); print('z=prior mean', mae_by_event(pm))
pm,_ = predict_zmean(R, swap=True); print('swap mu/lsd', mae_by_event(pm))
# bin edge rule <=
def vbin_le(code_name, x): return 1 + int(np.sum(np.array(med[code_name][2]) <= x))
```