Even the best gap input leaves errors of 0.03–0.15, so the gap term is not the problem; log(dt)/10 is about right. The error is uneven across bundles. Most are at 0.03–0.05, but two action bundles (k = 9 and 11, at hours 3.083 and 4.083) reach about 0.17. Something specific to those atoms is wrong, such as an unusual rate or dose, bins at an edge, or how END atoms are encoded.

Next I print this stay's records for hours 2 to 5 and the atoms I build for bundles 9 and 11. If those bundles hold, for example, rates exactly on a bin edge or an END atom whose attributes differ from its start, that points to the bug. If nothing stands out, the residual is spread through the computation.

```python
print(ev0[(ev0.hour>=2)&(ev0.hour<5.5)].to_string())
for k in [5,7,9,11]:
    print(k, bbB[k][0], [(a['code'],a['kind'],a['ek'],a['dose'],a['rate']) for a in bbB[k][1]])
for k in [9,11]:
    zt=torch.tensor(X0[k,960:]); zp=zk(k,math.log(300)/10)[0]
    print(k,'largest dim errs', sorted([(round(float(abs(zp[j]-zt[j])),3),j) for j in range(32)])[-5:])
```