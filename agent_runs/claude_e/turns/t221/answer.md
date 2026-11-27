The index error says `A_` does not line up with `datasets` the way I assumed: the row offsets ran past its 27,963 rows. So `datasets` may now hold more than 5 entries or a different set. Rather than guess, I'll print how many datasets there are, their row counts and which states file each came from. Then I can take the insulin-in-previous-bundle flag straight from each states file through the `prev` index, as I did for sel1.

```python
print('n datasets', len(datasets), [len(d[1]) for d in datasets], 'sum', sum(len(d[1]) for d in datasets), '| A_ rows', len(A_))
for i, (Xs_i, Rr_i) in enumerate(datasets):
    print(i, Xs_i.shape, Rr_i['file'].unique()[:3] if 'file' in Rr_i else None, Rr_i.stay.nunique(), int(Rr_i['prev'].max()))
```