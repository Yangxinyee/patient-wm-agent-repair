My head reconstruction from states matches the tool's predicted means only to about 1.3 (mn0 MAD). Some of that gap probably comes from the 4 random prior-z draws the tool averages over. It is good enough for diagnosis, not for exact offline checks. In earlier sessions I also built kernel-implied true shifts for the 100 tuning stays (TR/m, column 'true') to set beside the model's fwd_1 shifts. Next I'll print, for each edit, the mean kernel-implied true shift against the original model's shift. Those are the targets the edit has to reach. A true insulin x2 effect much larger than -2 would confirm that the model underestimates the insulin effect, as my notes suspected.

```python
print(m.columns.tolist(), m.shape)
g_ = m.groupby('edit').agg(true=('true','mean'), model=('shift','mean'), n=('shift','size'))
print(g_.round(3))
print(TR.groupby('edit')['true'].mean().round(3).to_dict())
print('K_ins', np.round(K_ins,2)); print('K_ne', np.round(K_ne,2))
print('K_ins_s', np.round(K_ins_s,2)); print('K_ne_s', np.round(K_ne_s,2))
```