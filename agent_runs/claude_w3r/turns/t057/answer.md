The two bad bundles both contain an insulin start at 0.50 u/h, and their errors sit almost entirely in z dimensions 9 and 20, at about 2.0 each. Bundle 3, with insulin at 1.0 u/h, matched well. So the bin assignment of some rates is probably wrong. My guess is that 0.5 lies exactly on a bin edge, and the model counts edges ≤ x where I count edges < x. If so, switching to side='right' for exact ties should remove the large errors.

Next I print the insulin rate and dose edges and redo the z comparison with the alternative tie rule.

```python
for (c,a_),d in NUM.items():
    if c in ('insulin','norepinephrine'): print(c,a_,'median',d['median'],'iqr',d['iqr'],'edges',np.round(d['fine_bin_edges'],4))
def fbin_r(code, attr, x):
    d=NUM[(code,attr)]; return 1+int(np.searchsorted(np.array(d['fine_bin_edges']), x, side='right'))
fbin_left = fbin
fbin = fbin_r
r_r = zerr3(UB,'base'); print('side=right', r_r)
for k in [3,9,11]:
    zt=torch.tensor(X0[k,960:]); zp=zk(k,math.log((bbB[k][0]-bbB[k-1][0])*3600)/10)[0]; print(k, round(float((zp-zt).abs().mean()),4))
fbin = fbin_left
```