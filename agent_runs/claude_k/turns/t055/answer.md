The context part looks right: the prior mean computed from my context is already closer to the saved z (error 0.091) than my posterior is (0.114). Of the input orders, [ctx, e, u] is clearly best, and adding my e_k does help. So the error must come from my measurement-bundle encoding e_k being wrong.

The most likely cause is a missing embedding term. Several categorical tables (unit, status, route, action, discharged_to, rate_unit), plus `embed.kq` and `embed.text`, could add a constant to every atom through a row-0 "absent" entry.

This run prints the row norms of all the small embedding tables. A non-zero row 0 marks a term that every atom receives. It then tests the plausible additions and some numeric-feature variants against the saved posterior means.

If one variant cuts the error to near zero, the measurement encoder is solved and I can move on to the action encoder.

```python
for t_ in ['embed.categorical.action.weight','embed.categorical.discharged_to.weight','embed.categorical.rate_unit.weight','embed.categorical.route.weight','embed.categorical.status.weight','embed.categorical.unit.weight','embed.kind.weight','embed.ekind.weight','embed.kq.weight','embed.text.weight','embed.bins.dose.weight','embed.bins.rate.weight']:
    print(t_, np.round(np.linalg.norm(W[t_], axis=1), 2))
def zerr_fn(make_vec, use_prior_ctx=True):
    errs = []
    for idx_, ip, dt, g_ in cand:
        vecs = [make_vec(e_, v_) for e_, v_ in zip(g_.event, g_.value)]
        e_k = set_enc(vecs, 'encode')
        zp = mlp(np.concatenate([ctx_of(ip, dt), e_k, np.zeros(192)]), 'z_post')[:32]
        errs.append(np.abs(zp - X[idx_, 960:992]).mean())
    return np.mean(errs)
base_mk = lambda e_, v_, **kw: atom_vec(CODE[e_], unit_row=3 if 'glucose' in e_ else 2, vbin=fbin(e_, 'value', v_), zv=zsc(e_, 'value', v_), hv=1.0, **kw)
tests = {
  'base': lambda e_, v_: base_mk(e_, v_),
  '+kq0': lambda e_, v_: base_mk(e_, v_, extra=W['embed.kq.weight'][0]),
  '+kq1': lambda e_, v_: base_mk(e_, v_, extra=W['embed.kq.weight'][1]),
  '+text0': lambda e_, v_: base_mk(e_, v_, extra=W['embed.text.weight'][0]),
  '+status0': lambda e_, v_: base_mk(e_, v_, extra=W['embed.categorical.status.weight'][0]),
  'hv=0': lambda e_, v_: atom_vec(CODE[e_], unit_row=3 if 'glucose' in e_ else 2, vbin=fbin(e_, 'value', v_), zv=zsc(e_, 'value', v_), hv=0.0),
  'no numeric bias': lambda e_, v_: base_mk(e_, v_) - W['embed.numeric.bias'],
}
for k_, f_ in tests.items(): print('%-16s %.4f' % (k_, zerr_fn(f_)))
```