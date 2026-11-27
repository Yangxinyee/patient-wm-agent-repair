GELU is confirmed. With all MLPs switched, the posterior correlation rose from 0.29 to 0.68 and its mean error halved to 0.060 (the saved values average 0.13). The prior's correlation also rose to 0.58, and the head's error dropped from 3.06 to 2.58 with no fitting.

The fit is still not exact, though. My unit-table rows for glucose and MAP (3 and 2) were chosen under the wrong activation, so I am re-searching them with GELU.

This run tries every assignment of unit rows to the two measurement units, using the posterior error as the score. It also tries two small encoder details: an unclipped z-score, and a "has value" flag of zero.

If one choice brings the posterior error to near zero (well under 0.01), the measurement encoder is exactly right. If the error stays around 0.05, the remaining gap is in the context or the core state, and I will look there next.

```python
def post_err(mk):
    P_ = np.array([mlp(np.concatenate([c_, set_enc([mk(e_, v_) for e_, v_ in zip(g_.event, g_.value)], 'encode'), np.zeros(192)]), 'z_post')[:32]
                   for (c_, _, _), (idx_, ip, dt, g_) in zip(pre_c, cand2)])
    return np.abs(P_ - T_all).mean(), np.mean([np.corrcoef(P_[:, j], T_all[:, j])[0, 1] for j in range(32)])
res_ = []
for ug in range(6):
    for um in range(6):
        mk = lambda e_, v_, ug=ug, um=um: atom_vec(CODE[e_], unit_row=(ug if 'glucose' in e_ else um), vbin=fbin(e_, 'value', v_), zv=zsc(e_, 'value', v_), hv=1.0)
        res_.append(post_err(mk) + (ug, um))
res_.sort(); print(res_[:5])
ug, um = res_[0][2], res_[0][3]
mk_noclip = lambda e_, v_: atom_vec(CODE[e_], unit_row=(ug if 'glucose' in e_ else um), vbin=fbin(e_, 'value', v_), zv=(v_ - NUM[(e_, 'value')]['median']) / NUM[(e_, 'value')]['iqr'], hv=1.0)
mk_hv0 = lambda e_, v_: atom_vec(CODE[e_], unit_row=(ug if 'glucose' in e_ else um), vbin=fbin(e_, 'value', v_), zv=zsc(e_, 'value', v_), hv=0.0)
print('noclip', post_err(mk_noclip), 'hv0', post_err(mk_hv0))
```