`RB` maps each reading bundle's row (2552 of the 5244 rows in the states file) to its readings, and `e_read(row)` returns that bundle's e in 1 ms. So I can rebuild the prelude at reading bundles. At drug bundles, e is the encoding of the END atoms of the infusion records ending in the same minute, not necessarily `e_empty()`. An error there would bias the reconstructed slot inputs and therefore the gate and decoders. I'll check whether one of my earlier helpers (`err_end`, `enc_e_var`, `efn`, `comp`) already encodes END-atom bundles and how their errors compared. Then I'll build the full reconstruction in the following run.

```python
for f in ['err_end','enc_e_var','efn','comp','end_atom','e_empty']:
    co=globals()[f].__code__
    print(f, inspect.signature(globals()[f]), [n for n in co.co_names if n in globals()][:12], [c for c in co.co_consts if isinstance(c,str)][:6])
```