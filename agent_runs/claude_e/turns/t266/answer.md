Pooled over all 400 tuning stays, the −2.6 edit is short of the truth estimate:

| Readout | Edited | Truth | Ratio |
|---|---|---|---|
| insulin x2 | −12.36 | −13.55 | 0.91 |
| insulin x1/2 | +5.84 | +6.77 | 0.86 |
| NE x2 | −0.23 | +4.53 | — |
| NE x1/2 | −0.56 | −2.26 | — |

f3, the batch I calibrated on, happened to be favourable; the other three batches ran 0.84–0.96 for x2. The readout part of the response scales linearly with the gain (from f3: −2.5 → −2.6 gave +4%). So matching x2 on average needs a gain of about −2.85 to −2.9, and −2.8 should give about 0.98–0.99 for x2 and about 0.92 for x1/2.

The cost of that is uncertain, because my per-0.1 cost factor of about 1.09 comes only from offline numbers. Two checks follow. First, I re-run the stay resampling with the costs scaled to −2.7, −2.75 and −2.8, to see which gains are even plausible. Then I build and save `ins_only(-2.8)` and measure it directly on tu0, the costliest batch (0.00428 at −2.6). If tu0 stays at or below about 0.005, I measure −2.8 on the other batches. If not, I back off.

```python
for fac,lab in [(1.0,'-2.6'),(1.095,'-2.7'),(1.148,'-2.75'),(1.20,'-2.8'),(1.30,'-2.9')]:
    out_=[]
    for N in [100,200,400,1000]:
        ii=rng.integers(0,len(PS8),size=(6000,N)); c=(v26*fac)[ii].sum(1)/nn[ii].sum(1); out_.append((N,round(float((c>.005).mean()),4)))
    print(lab,'mean',round(float((v26*fac).sum()/nn.sum()),5),out_)
E28=ins_only(-2.8)
for k in ['W0','b0']:
    print(k,'same as saved',float(np.abs(to_np(E28[k])-load(KMAP[k]).numpy().astype(np.float64)).max()))
print('TR untouched',float(np.abs(to_np(E28['TR'])-to_np(ORIG['TR'])).max()))
save_edit(KMAP['W2'], torch.tensor(to_np(E28['W2']),dtype=torch.float32))
print('W2 saved check',float(np.abs(load(KMAP['W2']).numpy().astype(np.float64)-to_np(E28['W2'])).max()), edited())
```