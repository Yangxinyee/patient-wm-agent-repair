Several helpers survive in the kernel. I will check the three pieces I need before rebuilding the edit on my own small, transparent code:
- with_code for a given z, by comparing wc_from at the prior-mean z with WC0;
- that HI['ob'] is the observed bin, so that minus log P0 of that bin gives nll0;
- what aug_wc returns.

If all three hold, I can compute the 4-draw factual NLL offline for any candidate edit, in both fp32 and bf16.

```python
w50 = np.asarray(wc_from(HI['zmu'][:50], HI['ctx'][:50], HI['reg'][:50]))
print('wc_from(zmu) vs WC0 maxdiff %.2e' % np.abs(w50 - WC0[:50]).max(), w50.shape)
ob_ = HI['ob'].astype(int)
nl_ = -np.log(np.clip(P0[np.arange(len(ob_)), ob_], 1e-12, None))
print('ob range', ob_.min(), ob_.max(), ' nll from P0[ob] vs nll0: rms %.4f, mean %.4f vs %.4f' % (np.sqrt(np.nanmean((nl_-rd_.nll0)**2)), np.nanmean(nl_), np.nanmean(rd_.nll0)))
try:
    ta_ = aug_wc(np.arange(10), 2, 0)
    print('aug_wc out', type(ta_).__name__, [getattr(x_, 'shape', None) for x_ in (ta_ if isinstance(ta_, (tuple, list)) else [ta_])])
except Exception as ex: print('aug_wc ERR', ex)
print('P0 is 4-draw avg?', P0.shape, 'zls range', np.round(HI['zls'].min(),2), np.round(HI['zls'].max(),2))
```