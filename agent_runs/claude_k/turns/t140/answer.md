The value-head layers match exactly (pre-activations reproduced to 0.0), and `pmall` is the tool's predicted mean. My stored `with_code` vectors (WC99) still give means about 2 mg/dL off the tool's. The error is systematic: glucose is about +1.7 too high and MAP about −0.5 to −1 too low.

Two likely causes:
- I built WC99 with an older guess of the 8-input layout of `last_numeric` (my notes mention a better fitted layout, A2, with MAE 1.3).
- I used a single z, whereas the tool averages the probabilities over 4 draws of z from the prior. The head is nonlinear, so averaging can shift the mean.

The next run tests which arrays carry these pieces: `mA2`/`mB` against the tool, whether `BASE99` equals WC99 − Q99 (the code-plus-register part), and which tensors build the view (`z_head`, `gap`, `z_prior`, `out_norm`).

If `mA2` is clearly better and z-averaging removes the bias, I'll rebuild with_code that way. That matters because the probe and the cost check should run on inputs close to the real ones.

```python
for nm_,arr in [('mA2',mA2),('mB',mB)]:
    print(nm_, 'MAE vs tool', np.abs(arr-pmall).mean(), 'bias by event', pd.Series(arr-pmall).groupby(R99.event.values).mean().round(2).to_dict())
print('WC-Q vs BASE99 maxdiff', np.abs((WC99-Q99)-BASE99).max(), ' |BASE99| mean', np.abs(BASE99).mean())
print([(n,SHAPES[n]) for n in NAMES if any(k in n for k in ['z_head','gap','z_prior','out_norm','flow','embed.code','embed.bins'])])
print('zp_ sample', np.round(zp_[0,:4],3), np.round(zp_[0,32:36],3), ' zm_ vs zp_ mean', np.abs(zm_-zp_[:,:32]).max())
print('fa4', np.round(fa4[:3],3)); print('M4', np.round(M4[:2],2)); print('A2', np.round(A2,3))
```