The kernel still holds the tuning-stay dose and reading grids (INS_U, NOR_U, GF_U, MA_U and sid2i), the estimated insulin lag profile (beta_ins) and many helper functions. Their sources can't be read, so I will check the signatures of the ones I may reuse.

Next I will break forward pass 8 down by hour and by lag. For each edit, I will regress the per-stay-hour predicted shift on the changed dose at lags 1 to 8. If the channel were linear, x2 and x1/2 would give the same lag coefficients, and those coefficients should match the profile estimated from the records. If the coefficients differ between x2 and x1/2, the channel is nonlinear. If lag 1 matches but later lags fall short, the effect is not being carried forward.

```python
import inspect, pandas as pd, numpy as np
for fn_ in ['true_shift_ins','true_shift_ne','true_shift','true_sh','tgt_rows','irf','dose_panel','build2','eval_edit','exp_tilt2','ridge_gate','fit_tilt']:
    try: print(fn_, inspect.signature(globals()[fn_]))
    except Exception as ex: print(fn_, 'ERR', ex)
print('g_lin', np.round(g_lin,2)); print('gam', np.round(gam,2)); print('beta_ins', np.round(beta_ins,2))
d8e = pd.read_csv('/work/fwd_8/dose_edit_edited.csv'); d8o = pd.read_csv('/work/fwd_8/dose_edit_original.csv')
mult = {'insulin_x2':2.0,'insulin_half':0.5,'ne_x2':2.0,'ne_half':0.5}
bne = np.zeros(13); bne[1:9] = g_lin
def lagD(df, K=8):
    L = np.zeros((len(df), K+1))
    for r_i, (sid_, h_, ed_) in enumerate(zip(df.stay_id.values, df.hour.values, df.edit.values)):
        D = INS_U if ed_.startswith('ins') else NOR_U
        i_ = sid2i[sid_]
        for k_ in range(1, K+1):
            j_ = h_ - k_
            if j_ >= 12: L[r_i, k_] = (mult[ed_]-1)*D[i_, j_]
    return L
L8 = lagD(d8e)
tg = np.where(d8e.edit.str.startswith('ins'), L8 @ beta_ins[:9], L8 @ bne[:9])
d8e['tgt'] = tg; d8e['orig'] = d8o['shift'].values
for ed_ in mult:
    mk = (d8e.edit==ed_) & d8e.reached
    sub_ = d8e[mk]
    print(ed_, 'mean tgt %.2f orig %.2f edit %.2f' % (sub_.tgt.mean(), sub_.orig.mean(), sub_['shift'].mean()))
    byh = sub_.groupby('hour')[['tgt','orig','shift']].mean().round(1)
    print(' h:', list(byh.index)); print(' tgt', list(byh.tgt)); print(' org', list(byh.orig)); print(' edt', list(byh['shift']))
    Lm = L8[mk.values][:, 1:7]
    for nm__, y_ in [('orig', sub_.orig.values), ('edit', sub_['shift'].values)]:
        cf = np.linalg.lstsq(Lm, y_, rcond=None)[0]
        print('  lagcoef', nm__, np.round(cf, 2))
```