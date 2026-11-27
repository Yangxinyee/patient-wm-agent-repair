The vocab defines z as (x − median)/IQR clipped to [−10, 10], and fine bin = 1 + the number of edges below x. Next I'll print the code table (ids, event kinds, units, action flag) and the numeric table for rate and dose. Those fix the atom embedding indices.

```python
for c_ in VOCAB['codes']:
    print(c_)
for n_ in VOCAB['numeric']:
    print(n_['code'], n_['attribute'], n_['median'], n_['iqr'], len(n_['fine_bin_edges']), n_['fine_bin_edges'][:4], '...', n_['fine_bin_edges'][-3:])
```