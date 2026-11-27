Offline the takeover works:
- At drug bundles z gets ≥ 0.98 of head-1 attention in 99% of cases (minimum 0.90).
- The other heads give z at most 0.0002.
- At reading bundles z gets at most 0.001, so non-drug bundles are effectively untouched.
- The head-1 output changes by a median of 9% (90th percentile 24%), mostly the intended innovation signal.

The "gain" statistic (−0.586) was a bug in my check. I clamped the denominator with `max(..., 1e-9)`, which wrecks every positive-innovation case, so the number means nothing.

In one run I'll:
1. Measure the gain properly by finite differences: change the insulin (or NE) innovation dim by ±1 and project the slot-stream change onto the glucose (or MAP) direction. It should be positive, around 0.3 per mg/dL-eq, given the 0.58 fit cosine.
2. Save the five edited tensors.
3. Print the stays of the earlier dose_edit run, for which I have per-stay-hour targets.

A clearly positive gain confirms the signs. A negative one means I must flip the z_in columns before testing.

```python
gs={'G':[], 'M':[]}
for k in kd[::8]:
    i4=kd.index(k); e,u=[np.asarray(x,float) for x in enc_e_var((int(sid[k]),round(float(hr[k]),3)),'none')]
    hp=xo[k-1,192:960].reshape(4,192)*np.exp(-(hr[k]-hr[k-1])*3600*np.exp(fa)); z0=zmean_new(ctxs[i4],e,u)
    for nm,dim,D,scale in [('G',ZD[4],DG,-kA),('M',ZD[5],DM,kB)]:
        x_=[]
        for dlt in (-0.5,0.5):
            z1=z0.copy(); z1[dim]+=dlt/abs(scale); x_.append(x1_of(hp,e,u,Zwn@z1+Zb).mean(0))
        gs[nm].append((x_[1]-x_[0])@D/(D@D)/np.sign(scale))   # per 1 unit of eps (mg/dL-eq or mmHg-eq), along +reading direction
print('gain per eps along reading dir: G med %.3f [%.3f,%.3f]  M med %.3f [%.3f,%.3f]'%(np.median(gs['G']),*np.percentile(gs['G'],[10,90]),np.median(gs['M']),*np.percentile(gs['M'],[10,90])))
save_edit('z_post.0.weight',W0n.astype(np.float32)); save_edit('z_post.0.bias',b0n.astype(np.float32))
save_edit('z_post.2.weight',W2n.astype(np.float32)); save_edit('z_post.2.bias',b2n.astype(np.float32)); save_edit('z_in.weight',Zwn.astype(np.float32))
print(edited())
fwst=sorted(fw.stay_id.unique().tolist()); print(len(fwst), json.dumps([int(s) for s in fwst]))
```