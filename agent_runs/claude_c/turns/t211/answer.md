Pass 22 saved only the edited model's factual output. The original's per-reading output on the same 100 tuning stays should be the kernel's `fu` (4751 rows). I'll confirm that by checking its mean NLL is 1.63461. Then I'll split the head-0 ×4 NLL change by event and by time since the last drug record.

```python
tu = pd.read_csv('/data/tuning.csv'); print(tu.shape, list(tu.columns)[:12])
print('fu mean nll', round(fu.nll.mean(),5), 'fe mean', round(fe_.nll.mean(),5))
m_ = fu.merge(fe_, on=['stay_id','hour','event','observed'], suffixes=('_o','_e')); m_['d']=m_.nll_e-m_.nll_o
print(len(m_)); print(m_.groupby('event').d.agg(['mean','count']).round(4).to_string())
```