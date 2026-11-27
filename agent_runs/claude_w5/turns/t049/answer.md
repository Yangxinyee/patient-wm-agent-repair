The vocab gives the z-score rule (x − median)/IQR, clipped to [−10, 10], and fine bins of 1 + the number of edges below the value. The value head predicts a mean as Σ p(b)·centre. Next I'll print all 8 numeric entries compactly and the remaining codes (ids 9–12), so the embedding can be rebuilt exactly.

```python
for e_ in num:
    print(e_['code'], e_['attribute'], e_['median'], e_['iqr'], [round(x,2) for x in e_['fine_bin_edges']])
print([ (c_['id'],c_['name']) for c_ in VOCAB['codes'][9:]])
print({k_:(len(v_) if hasattr(v_,'__len__') else v_) for k_,v_ in VOCAB['value_bin_centres'].items()} if isinstance(VOCAB['value_bin_centres'],dict) else len(VOCAB['value_bin_centres']))
```