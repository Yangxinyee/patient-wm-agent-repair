The full architecture shows there is no dose memory outside the slot state h:
- The memory M stores only measurement and visit bundle vectors, not drug encodings.
- The registers hold only the last value of the same code.

So effects at lag 3 and beyond must come either from new memory written into h or from the model carrying its own sampled readings forward. Writing new memory into h would mean editing the prelude, the recurrent blocks, c_out and the coda together, which is risky for the factual fit.

There is also a question the head-level design has to answer first. A linear analysis says that if the head corrected the lag-1 and lag-2 shortfalls as designed, the rollout should already be close to the truth. It is not, so either the lag-2 channel is not delivering in rollouts, or there is a feedback loop.

That feedback loop would work like this: the dose-surprise decoder centres the dose on the glucose-implied dose, and in a rollout that glucose comes from the model's own samples. If this is the cause, the edited model's factual arm (recorded doses) should drift away from the original model's factual arm over hours 13 to 24. If the factual arms agree, the loss is in propagation.

I will compare the two factual arms hour by hour, compare both against the real readings, and look at the per-stay spread of the shifts.

```python
gfu = {}
for sid_ in d8e.stay_id.unique():
    gfu[sid_] = GF_U[sid2i[sid_]]
for ed_ in ['insulin_x2','insulin_half','ne_x2','ne_half']:
    mk = (d8e.edit==ed_)
    a_ = d8e[mk]; b_ = d8o[mk.values]
    real_ = np.array([ (GF_U if ed_.startswith('ins') else MA_U)[sid2i[s_], h_] for s_, h_ in zip(a_.stay_id, a_.hour)])
    dfc = pd.DataFrame({'hour':a_.hour.values, 'fe':a_.factual.values, 'fo':b_.factual.values, 'real':real_, 'r':a_.reached.values,
                        'se':a_['shift'].values, 'so':b_['shift'].values})
    dfc = dfc[dfc.r]
    g_ = dfc.groupby('hour')
    print(ed_, 'factual arm edited-orig by hour:', list(g_.apply(lambda x: (x.fe-x.fo).mean()).round(2)))
    if ed_ in ('insulin_x2','ne_x2'):
        ok_r = ~np.isnan(dfc.real)
        print('  real-orig_fact by hour:', list(dfc[ok_r].groupby('hour').apply(lambda x: (x.real-x.fo).mean()).round(1)))
        print('  real-edit_fact by hour:', list(dfc[ok_r].groupby('hour').apply(lambda x: (x.real-x.fe).mean()).round(1)))
    q_ = dfc.groupby('hour').apply(lambda x: np.percentile(x.se - x.so, [10,50,90])).values
    print('  incr pct10/50/90 h13,h18,h24:', np.round(q_[0],1), np.round(q_[5],1), np.round(q_[-1],1))
```