No z dimension offers a usable linear dose channel. The low-variance dimensions that correlate with rate have slopes of about 0.004 per U/h, while their spread at reading bundles is 50–200× larger.

I now think the model's insulin response cannot be made linear in dose with these edits. But it may not need to be linear for the readout. The original pooled shifts are −1.85 (x2) and +0.78 (x1/2), against my estimated −13.1 and +6.5. These ratios, 7.1 and 8.4, are similar. Amplifying the model's existing dose-induced state change by about 7.5×, rather than its dose input, could bring both insulin readouts near the truth.

To do that I need the direction of that state change. Pass 5 already contains it: there the edited model differed from the original only in how dose is embedded, and both models ran on identical records. So the paired difference of states at insulin bundles isolates the dose-driven change, free of cross-stay noise. I take the change across each insulin bundle under the edit minus the same change under the original. I then check:
- how large that difference is relative to the state's natural variation;
- whether it lies along a few consistent directions;
- whether it scales with the recorded dose.

A low-rank, dose-proportional difference supports a direction-based amplification; a diffuse or tiny one argues against it.

```python
xo = So['x']; xe = Se['x']; sidx = So['stay_id']
same = np.r_[False, sidx[1:]==sidx[:-1]]
Dd = (xe - xo)                       # paired level difference after each bundle
DD = Dd.copy(); DD[same] = Dd[same] - Dd[np.r_[same[1:], False]][:0].sum() if False else Dd[same] - Dd[np.where(same)[0]-1]
kinds_arr = kinds
for k_ in ['ins','ne_only','read','other']:
    m_ = kinds_arr==k_
    print(k_, "mean |level diff| out %.4f h %.3f ; mean |change diff| out %.4f h %.3f" % (
        np.linalg.norm(Dd[m_,0:192],axis=1).mean(), np.linalg.norm(Dd[m_,192:960],axis=1).mean(),
        np.linalg.norm(DD[m_,0:192],axis=1).mean(), np.linalg.norm(DD[m_,192:960],axis=1).mean()))
print("natural sd of out change across ins bundles (norm):", np.linalg.norm((xo[ins_b[1:]]-xo[ins_b[1:]-1])[:,0:192] - (xo[ins_b[1:]]-xo[ins_b[1:]-1])[:,0:192].mean(0),axis=1).mean().round(3))
# structure of change-diff at insulin bundles without NE
mi = np.where((kinds_arr=='ins') & same & ~hn3)[0]
zi_rate = np.array([lk('insulin', s, int(np.floor(h))) for s,h in zip(sidx[mi], hr3[mi])])
zins = zi_rate - 1.5
Uo, so_sv, Vo = np.linalg.svd(DD[mi, 0:192], full_matrices=False)
print("out change-diff singular values (top 6):", (so_sv[:6]/np.sqrt(len(mi))).round(4), " frac var top1 %.2f top3 %.2f" % ((so_sv[:1]**2).sum()/(so_sv**2).sum(), (so_sv[:3]**2).sum()/(so_sv**2).sum()))
p1 = DD[mi, 0:192] @ Vo[0]
print(pd.DataFrame({'p1':p1, 'rate':zi_rate}).groupby('rate')['p1'].agg(['mean','std','count']).round(4).T.to_string())
```