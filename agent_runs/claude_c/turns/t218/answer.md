Head-0 ×4 does both things. It biases predictions systematically (glucose_lab +0.8, MAP about +0.3) and adds bundle-specific noise (glucose prediction sd 1.9). The shifts are uncorrelated with the residuals, so they are pure cost. This is not a fixable offset; whole-head amplification is non-selective.

My plan is now to build the best achievable combination of the selective input-side lever with a modest head-0 gain, within the factual budget. The selective lever is the z_dose and z_rate columns of `embed.numeric`, scaled ×5; it touches only drug atoms. Its factual cost has never been measured, so that comes first. I'll record the findings, clear the stale head-0 ops, and set up numeric ×5.

```python
txt_ = """
## Restart-9 (runs 155-167)
- Found ops = head0 x4 again (48 basis 'in' ops alpha 3 on prelude attn.out cols 0-47); purpose lost. Cleared.
- Head0 x4 factual cost (vs fwd_2 original) spread over all events & time-since-drug bins (incl >12h): gf +.006, gl +.018, art +.011;
  pred shifts: gl mean +0.80 sd 1.9, gf sd 1.9, MAP +0.3 bias; corr with residual ~0 -> non-selective noise+bias. Head scaling = dead end.
- normalize(h): 1 dominant dir (184.8/192 uncentred); variation 21 dims (99%), 40 (99.9%). kv/q/out have no bias; u-token at reading
  bundles = LN1(0) = norm1.bias. Selective LN1-space edit would need u-token geometry (unobservable) + would amplify drug-presence offset.
- Plan: modest combo. Step 1: factual cost of numeric z_dose/z_rate x5 (never measured).
"""
open('/work/notes.md','a').write(txt_)
clear_ops()
for j_ in [2,4]:
    e_ = np.zeros(6); e_[j_]=1.0; rank1('embed.numeric.weight', e_, 4.0, side='in')
print(ops())
```