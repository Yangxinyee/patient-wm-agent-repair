The alignment is fine: the stored z belongs to its own bundle (offset 0 fits best), so that is not why the rebuild fails.

The Forward-1 file has a `reached` flag marking the hours that count toward each readout. My stored causal kernels are 10 lags for insulin (sum −11.2) and 8 lags for norepinephrine (sum +9.2).

Next I will compute the true effect for every reached stay-hour of the 100 Forward-1 tuning stays and compare it with the original model's predicted shifts. That gives the target for each of the four readouts and the size of the gap. I will also write my current findings to the notebook. I expect the true insulin ×2 readout to be around −10 to −15 mg/dL, well beyond the model's −1.9. For norepinephrine ×2 I expect a small positive value, where the model gives the wrong sign.

```python
tu_i = tu[tu.event=='insulin'][['stay_id','hour','rate']].copy(); tu_i['hr'] = np.floor(tu_i.hour + 1e-6).astype(int)
tu_n = tu[tu.event=='norepinephrine'][['stay_id','hour','rate']].copy(); tu_n['hr'] = np.floor(tu_n.hour + 1e-6).astype(int)
rI = {(a, b): c for a, b, c in zip(tu_i.stay_id, tu_i.hr, tu_i.rate)}; rN = {(a, b): c for a, b, c in zip(tu_n.stay_id, tu_n.hr, tu_n.rate)}
mult = {'insulin_x2': 2.0, 'insulin_half': 0.5, 'norepinephrine_x2': 2.0, 'norepinephrine_half': 0.5}
print(d1.edit.unique())
def true_shift(sid_, t, ed):
    m_ = [v for k_, v in mult.items() if ed.startswith(k_.split('_')[0]) and ed.endswith(k_.split('_')[1])]
    m_ = m_[0] if m_ else (2.0 if 'x2' in ed else 0.5)
    if ed.startswith('insulin'): ps, rr = psi_ins, rI
    else: ps, rr = psi_ne, rN
    return (m_ - 1) * sum(ps[l-1] * rr.get((sid_, t - l), 0.0) for l in range(1, len(ps) + 1) if t - l >= 12)
d1['true'] = [true_shift(a, b, c) for a, b, c in zip(d1.stay_id, d1.hour, d1.edit)]
summ = d1[d1.reached].groupby('edit').agg(model=('shift', 'mean'), true=('true', 'mean'), n=('shift', 'size'))
print(summ.round(3).to_string())
print(d1[d1.reached & d1.edit.str.startswith('insulin_x2')].groupby('hour')[['shift', 'true']].mean().round(2).T.to_string())
with open('/work/notes.md', 'a') as fh:
    fh.write("""
## Runs 30-47 (after restart)
- Heads reimplemented & VALIDATED vs factual predicted means (rmse = 4-sample z noise): view = out + LN_outnorm(mean slots of h*exp(-dt_s*exp(flow.a))) + gap(log(dt_s)/10) + z_head(z~prior(ctx)); q = query_norm(view + type_emb[1]) (type 1 = WORLD measurement; types=kind*5+ekind?); with_code = q + code_emb + last_numeric(f) + bins.value[last vbin]; f: col0 has_last, col3 log1p(age_s)/10, col6 z_last_value, others 0; logits = value_bin MLP(gelu) + embed_transition.value_bin[code*17+lastbin]; mean = sum p[1:]*centre.
- Bundles = events grouped by minute; END atoms at infusion end (h+1.083) share bundle with next start; visit END atoms at 25.1667. Matches states (5369/5369).
- Encoder/z_post reimplementation FAILED (best mae 0.04 vs scale 0.10) despite searching numeric layout, unit rows, kind/kq, ctx decay. Gave up on core replica for now.
- Embedding rows ~0.277 norm = untrained (never used): kq1-3, text1, action1, route1, status1, discharged1, rate_unit1, unit1, kind1.
- Records: tier mean insulin rates 1.0/2.0/3.0/3.71. Regression G(t)~G lags(0.46,0.28,0.18)+tier dummies+dev lags: direct dev kernel k=[-2.1,-1.54,-0.53,+.35,+.73,+.70,+.70,+.55].
- Decodability from state (before glucose reading) of insulin dev: lag1 R2 0.27, lag2 0.07, lag3 0.11, lag4 ~0 -> state carries little lagged dose info.
""")
```