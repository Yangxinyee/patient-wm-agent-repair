No variant of the value encoding matches either: the best is still off by 0.47. Changing the bin rule, the numeric vector, the z scaling and the unit row cannot explain reading atoms. Yet the same machinery reproduces drug and END atoms exactly.

Before guessing further I'll look at the data directly:
- **Within one fine bin:** does e_k change with the reading value? If it does, a numeric value term is present.
- **Across a bin edge:** does it jump? That would show the bin term.
- **Which coordinates differ:** comparing the probe's e_k for a single map_cuff reading with my replica's prediction for the same reading shows how the residual is structured.

```python
d_c = num[('map_cuff','value')]; print('cuff edges', d_c['fine_bin_edges'][:8])
for v_ in [41.,42.,43.,44.,45.,46.,47.,48.,49.,51.,52.,53.]:
    if ('map_cuff', v_) in dd_: print(v_, 'bin', fbin(v_, d_c['fine_bin_edges']), 'n', len(dd_[('map_cuff', v_)]), 'e', np.round(dd_[('map_cuff', v_)][0], 3))
```