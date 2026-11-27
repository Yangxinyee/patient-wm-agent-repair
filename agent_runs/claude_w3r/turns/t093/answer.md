Run 86 printed my helper code and then failed: the edges are stored in `NUM[(code, attr)]['fine_bin_edges']`, not where I looked. My `fbin` uses side='left' (bin = 1 + number of edges below x).

Same test, corrected: compare the error on rows whose last glucose lies exactly on an edge, then recompute everything with side='right'.
- **Confirms the bin bug:** edge rows carry the excess error and side='right' brings the median down toward the 1.0 floor.
- **Refutes it:** nothing changes.

```python
ed = np.asarray(NUM[('glucose_finger','value')]['fine_bin_edges'], float); print('gf edges', ed)
gls = np.array([x[3] if x[3] is not None else np.nan for x in pre2])
on_edge = np.array([(g==g) and bool(np.any(np.isclose(g, ed))) for g in gls])
print('frac on edge', on_edge.mean().round(3))
if on_edge.any(): print('median|e| on-edge', np.median(np.abs(e_st[on_edge])).round(3), ' off-edge', np.median(np.abs(e_st[~on_edge])).round(3))
_fbin_left = fbin
def fbin(code, attr, x):
    d=NUM[(code,attr)]; return 1+int(np.searchsorted(np.array(d['fine_bin_edges']), x, side='right'))
a, b, e_r = score2(pre2); print('side=right:', round(a,3), round(b,3))
if on_edge.any(): print('  on-edge', np.median(np.abs(e_r[on_edge])).round(3), ' off-edge', np.median(np.abs(e_r[~on_edge])).round(3))
fbin = _fbin_left
```