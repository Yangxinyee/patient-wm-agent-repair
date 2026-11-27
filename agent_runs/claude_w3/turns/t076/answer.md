The dose_edit file has one row per stay-hour and edit. The columns are the model's predicted shift, its standard error, the factual predicted mean, and a `reached` flag (whether the drug had already run from hour 12 on). I will now apply my stored helper `readout_truth` to these rows for all four edits. It uses the impulse responses above and each stay's recorded doses. The result is the true readout on exactly the rows the tool averages over, and the model's current readout next to it. I expect about −17/+9 for insulin and small values for norepinephrine.

```python
res_t = {}
for ed_, D_, th_, mu_ in [('insulin_x2',INS_tu,th_ins,2.0),('insulin_x0.5',INS_tu,th_ins,0.5),('norepinephrine_x2',NE_tu,th_ne,2.0),('norepinephrine_x0.5',NE_tu,th_ne,0.5)]:
    try:
        r_ = readout_truth(de1, ed_, D_, th_, mu_)
        res_t[ed_] = r_
        print(ed_, np.round(r_,3) if not isinstance(r_, tuple) else [np.round(x,3) if np.ndim(x)==0 else np.shape(x) for x in r_])
    except Exception as e_:
        print(ed_, 'ERR', repr(e_)[:200])
print(de1.edit.unique())
print(de1[de1.reached].groupby('edit')['shift'].mean().round(3).to_dict())
```