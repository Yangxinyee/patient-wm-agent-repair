`encode_bundles(AT, rows)` reproduces `eALL` and `uALL` exactly and runs in 0.05 s for 5,369 bundles. I now have every head tensor's shape and the code indices (glucose_finger 6, map_arterial 7, insulin 8, norepinephrine 9, and so on).

To run rollouts on any stays, including the tuning stays used in forward 1, I need to build the atom table `AT` from the raw records. So I'll inspect the raw record frames (`tr`, `tu`) and one stay's full `AT` side by side, and recover the construction rules: END atoms, visit atoms, minutes and bundle rows. I'll also print the factual-output frame `R`, which I'll use to validate a batched head I'll write next. If the mapping is clear, the next run writes `make_atoms` and checks it reproduces `AT` exactly for the sel2 stays.

```python
print(open('/data/README.md').read()[:2500])
print("tr cols", list(tr.columns)); print(tr.head(3).to_string())
print("tu cols", list(tu.columns)); print(tu.head(3).to_string())
s0x = AT['stay_id'].iloc[0]
a0 = AT[AT.stay_id==s0x]
print(a0.head(25).to_string())
print("R cols", list(R.columns)); print(R.head(3).to_string())
```