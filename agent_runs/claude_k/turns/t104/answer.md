The arrays from runs 74–86 are there for the 99 state stays (5,521 glucose/MAP readings). They hold q, with_code, the head's pre-activations, GELU activations, logits, the transition bias, the observed bins, the codes, and an R99 table of readings. XS (20,230 × 992) looks like the states of every bundle.

So I can work offline. First I need to know what R99 contains and how faithful my head reconstruction is. I'll compare my recomputed logits, softmax over my logits plus the transition bias, against the tool's per-reading NLL and predicted mean. A close match (NLL difference about 0.01 or less, mean error about 1–2 units) would let me cost head edits offline. A poor match would mean I have to fix the reconstruction first.

```python
print(list(R99.columns))
print(R99.iloc[0].to_dict())
lg = lg99 + 0.0
def lsm(x): m = x.max(1, keepdims=True); return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))
for name, L in [('lg', lg99), ('lg+TR', lg99 + TR99)]:
    lp = lsm(L); nll = -lp[np.arange(len(ybin99)), ybin99]
    print(name, 'mean nll', nll.mean().round(4), 'vs base', base_nll.mean().round(4), 'mad', np.abs(nll - base_nll).mean().round(4))
cen = np.array(VOCAB['value_bin_centres']) if 'value_bin_centres' in VOCAB else None
print(type(cen), None if cen is None else cen.shape)
```