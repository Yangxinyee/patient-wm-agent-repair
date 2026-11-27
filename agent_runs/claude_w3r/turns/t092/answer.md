Run 85: measuring the register age at the state's time removes the mean bias (−0.14). But the median |error| is still 2.0, about twice the sampling floor, so another error remains. The earlier error distribution had heavy tails, which points to a discrete mismatch on a subset of rows rather than a smooth one. The prime suspect is the last-value bin index: the side convention when an integer glucose value sits exactly on a bin edge. That index sets both the bin embedding and the transition row.

Next I'll print my `fbin` source and the glucose_finger edges. Then I'll compare errors for rows whose last value lies on an edge against the rest, and test the other side convention. If edge rows carry most of the error and the other convention fixes them, I have the bug. If the errors are spread evenly, the cause is elsewhere, for example the z-prior scale.

```python
print(src)
ed = np.asarray(V['bin_edges']['glucose_finger']['value'] if 'bin_edges' in V else None, dtype=float) if False else None
import inspect
def edges_of(code, attr):
    for k in ['bin_edges','edges','bins']:
        if k in V:
            t = V[k]
            try: return np.asarray(t[code][attr], float)
            except Exception: pass
    return None
ed = edges_of('glucose_finger','value'); print('gf edges', ed)
gls = np.array([x[3] if x[3] is not None else np.nan for x in pre2])
on_edge = np.array([ (g==g) and np.any(np.isclose(g, ed)) for g in gls])
print('frac on edge', on_edge.mean().round(3))
print('median|e| on-edge', np.median(np.abs(e_st[on_edge])).round(3) if on_edge.any() else None, ' off-edge', np.median(np.abs(e_st[~on_edge])).round(3))
```