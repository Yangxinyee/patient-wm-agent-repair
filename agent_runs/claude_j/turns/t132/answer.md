`BUN` is keyed by integers, not strings, hence the KeyError. The bytecode shows how things fit together. `one_step_err` takes the tool's state rows `Xt` (992 = out 192 | h 4×192 | z 32), turns each bundle into vectors with `bvd`, and checks `step` against the next row. So the replica chain is usable. My first goal is cheaper and doesn't need counterfactual states: rebuild the reading table and head query vectors q from the stored tool states, plus the dose-history targets. Then I can fit a linear readout. I'll call the existing helpers on the cached data and print only shapes and column names.

```python
k0_=list(BUN.keys())[0]; print(type(k0_), type(BUN[k0_]), len(BUN[k0_]))
print('S keys:', list(S.keys())[:8])
try:
    RTz=reading_table(S, BUN); print('RTz', type(RTz).__name__, getattr(RTz,'shape',None), list(RTz.columns)[:20] if hasattr(RTz,'columns') else '')
except Exception as ex_: print('reading_table err', type(ex_).__name__, str(ex_)[:200])
print('RT type', type(RT).__name__, 'R2 cols', list(R2.columns) if hasattr(R2,'columns') else R2[:1])
print('tu cols', list(tu.columns) if hasattr(tu,'columns') else '', 'tr cols', list(tr.columns) if hasattr(tr,'columns') else '')
```