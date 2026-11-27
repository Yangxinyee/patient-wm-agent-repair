The piecewise-linear corrections help only modestly: at best 1.3-1.5 times the response per unit cost, and never on both edits at once. That's not worth the fragility.

The tool's cost per glucose_finger row (0.0062) is about twice my in-sample estimate (0.0029). That suggests the insulin readout is much noisier on new stays than on the stays it was fit on, and that better regularisation or more training data may be the bigger lever.

I can measure its out-of-sample quality without a new pass. In fwd_11 the edited minus original predicted mean on gated glucose rows is about -1.9 times the readout, so I can recover the readout on 100 unseen tuning stays. I'll compare it with the true dose deviation computed from the records, using the sliding-scale tier of the previous finger reading. I'll also check that held-out head inputs (WC, sel1 rows) still exist in the kernel for later readout tests.

```python
fo = pd.read_csv('/work/fwd_11/factual_original.csv'); fe = pd.read_csv('/work/fwd_11/factual_edited.csv')
ff = fo.merge(fe, on=['stay_id','hour','event','observed'], suffixes=('_o','_e'))
ff['sh'] = ff.predicted_mean_e - ff.predicted_mean_o; ff['dnll'] = ff.nll_e - ff.nll_o
gfG = Gtu['gf']; insT = Gtu['ins']
Etab = {0:0.0, 1:1.03, 2:2.07, 3:3.24, 4:6.24}
def tier_val(v):
    if pd.isna(v): return None
    return 0 if v<150 else 1 if v<180 else 2 if v<220 else 3 if v<300 else 4
g = ff[ff.event=='glucose_finger'].copy()
g['d'] = [insT.at[s,h-1] if (h-1) in insT.columns and pd.notna(insT.at[s,h-1]) else 0.0 for s,h in zip(g.stay_id, g.hour)]
g['tr'] = [tier_val(gfG.at[s,h-1]) if (h-1) in gfG.columns else None for s,h in zip(g.stay_id, g.hour)]
g['dev'] = [d - (Etab[t] if t is not None else 0.03) for d,t in zip(g.d, g.tr)]
on = g.d > 0
ru = g.sh[on]/(-1.9)
print('gf rows %d, with insulin prev hour %d' % (len(g), on.sum()))
print('out-of-sample: corr(readout, dev) %.3f  R2 %.3f  slope readout-on-dev %.3f  readout sd %.3f dev sd %.3f' % (np.corrcoef(ru, g.dev[on])[0,1], np.corrcoef(ru, g.dev[on])[0,1]**2, np.cov(ru, g.dev[on])[0,1]/g.dev[on].var(), ru.std(), g.dev[on].std()))
print('rows without insulin: mean |shift| %.3f  (gate leakage)' % g.sh[~on].abs().mean())
print('dNLL on gf: insulin rows %.4f, other rows %.4f' % (g.dnll[on].mean(), g.dnll[~on].mean()))
print('WC' in globals() and np.shape(WC), 'RR' in globals() and RR.shape)
```