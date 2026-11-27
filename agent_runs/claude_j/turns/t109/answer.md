Two things follow. The zero-bin embeddings contribute nothing, which is consistent with them being zero padding rows. Age is measured in seconds, since minutes and hours are clearly worse.

First readings still miss by 2.3 on average, against 0.94 for the others. Some input specific to "no previous value" may not be zero. For example, the age might count from admission rather than being absent, or the error might sit in one code.

I will break the first-reading error down by code and by bundle position. I will then try filling the age field with time since admission, or since the hospital-admission bundle, with has_last kept at 0. If one variant brings the error to about 1, that settles the register semantics.

```python
d1 = (head_mean(view0_, 1, Fn.gelu, ext_for(), tb=lb_)[0] - tool_)
RTb['err0'] = d1.numpy()
print(RTb[RTb.has_last == 0].groupby('event')['err0'].agg(['count', 'mean', lambda x: np.abs(x).mean()]).round(2).to_string())
print(RTb[RTb.has_last == 0].assign(bb=lambda d: np.minimum(d.b, 4)).groupby('bb')['err0'].agg(['count', lambda x: np.abs(x).mean()]).round(2).to_string())
# variant: for first readings, age = time since first bundle of stay (admission) in seconds
adm_min = {s: BUN[s][0][0] for s in BUN}
age_adm = np.array([(m - adm_min[s]) * 60.0 for s, m in zip(RTb.stay_id, RTb.minute)])
for nm_v, ages_first, hflag in [('age_adm_h0', age_adm, 0.0), ('age_adm_h1', age_adm, 1.0)]:
    e_ = ext_for()
    f = (hl_ == 0)
    ag2 = torch.tensor(np.log1p(ages_first) / 10.0)
    e_[f] = e_[f] + ag2[f][:, None] * LW[:, 3] + hflag * LW[:, 0]
    mu_, _ = head_mean(view0_, 1, Fn.gelu, e_, tb=lb_)
    print(nm_v, 'first MAE %.3f' % float((mu_ - tool_)[m1].abs().mean()))
```