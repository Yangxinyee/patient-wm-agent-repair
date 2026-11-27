The linear decoder reaches R² 0.95, but that is not clean enough. At insulin bundles its error (sd 0.49) is as large as the within-tier deviation signal itself. Reading bundles add errors of sd 0.29 and all other bundles 0.19 per write. Accumulated in a persistent memory, that error would inject spurious glucose shifts of about 1 mg/dL. That would roughly cancel the factual gain from the true dose deviations.

Before redesigning, I want to locate the limit. Either the rate information in E is fundamentally fuzzy, which is a problem of attention mixing in the prelude, or one shared linear row simply cannot serve all bundle types at once.

**Next step:** three diagnostics.
1. The best-case decodability of the rate within insulin bundles alone.
2. The prelude attention weights from the slots to the u (dose) token at insulin bundles, per head. A head that consistently attends to u would give a clean route for the write.
3. The size of the within-tier deviation signal I need to carry.

**How I will decide:**
- If (1) is ≈0.99, the noise comes from sharing one row across bundle types, and better target design or extra features can fix it.
- If (1) is also poor, the dose information must be routed more cleanly, through a dedicated attention path to the u token.

```python
ins_m = btype=='ins'
rg1 = Ridge(alpha=1e-3, fit_intercept=False).fit(Eb[tr_m&ins_m], yI[tr_m&ins_m])
rr = yI[te_m&ins_m]-rg1.predict(Eb[te_m&ins_m]); print("ins-only decoder: test R2 %.4f resid sd %.3f"%(1-rr.var()/yI[te_m&ins_m].var(), rr.std()))
# within-tier deviation of insulin rate (vs tier of preceding glucose reading)
prev_row = dsA['bd']['prev'].values
dev_ = np.array([yI[r] + yI[prev_row[r]] if (prev_row[r]>=0 and btype[prev_row[r]]=='gread') else np.nan for r in np.where(ins_m)[0]])
print("within-tier dev sd %.3f (n=%d, nan %d)"%(np.nanstd(dev_), len(dev_), np.isnan(dev_).sum()))
# attention weights slots->u token at insulin bundles
AW = torch.zeros(nbA, 4, 7, 7)
for k in range(len(CAP)):
    act_ = np.where(lens > k)[0]
    for pos, s_i in enumerate(act_):
        AW[dsA['srows'][s_i][k]] = CAP[k][2]['aw'][pos]
aw_ins = AW[torch.tensor(ins_m)]          # n x heads x 7(q) x 7(k)
print("mean attn slots->u by head:", [round(float(aw_ins[:,hh,:4,5].mean()),3) for hh in range(4)], " sd:", [round(float(aw_ins[:,hh,:4,5].std()),3) for hh in range(4)])
print("per head/slot mean:", np.round(aw_ins[:,:,:4,5].mean(0).numpy(),2).tolist())
```