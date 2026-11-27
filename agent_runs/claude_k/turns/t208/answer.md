This settles where the signal goes and where it is lost:
- A z write does move the recurrent memory. Each h slot shifts by about 5 per unit of dz18 and 3 per unit of dz28, the same in all four slots. That is several percent of h's natural spread (SD-norm 76).
- `out` and ctx barely move (about 1–2% of their spread). Both are LayerNorm'd summaries of the large raw slot mean, so a shift that is sizeable in h becomes small after normalisation. The heads read the same normalised view, so the value head sees the same tiny change.

So the real question is not the size of Δq but its signal-to-noise. Does the dose-driven Δq at later readings point along directions where q's natural variance is small?
- If yes, a readout in the value head can amplify it cheaply, and the z channel (possibly with larger writes) is workable.
- If Δq lies along high-variance directions, any readout amplifies natural noise as much as the dose signal, and the z channel is a dead end.

To measure this I need q at reading bundles for both fwd8 models. The kernel has `qrows` and per-reading metadata, but I don't remember their columns. So this run identifies how to rebuild q for a reading (previous-state row index and gap). It also checks my q rebuild against the stored q for the 300-stay set before I rely on it.

```python
print('R cols', list(R.columns)); print(R.head(2).to_string())
R5=D5['R']; print('D5 R cols', list(R5.columns)); print(R5.head(2).to_string())
print('D5 kk', D5['kk'][:8], 'code', D5['code'][:8], 'lb', D5['lb'][:8])
import inspect
print('qrows sig', inspect.signature(qrows), ' view_from sig', inspect.signature(view_from))
# try rebuild of q for first 50 readings of D5 using S5 states
kk5=D5['kk']; 
try:
    dts5=R5['dt'].values if 'dt' in R5.columns else None
    print('dt col present', dts5 is not None)
except Exception as ex: print(ex)
```