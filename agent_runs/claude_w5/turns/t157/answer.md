My prelude reimplementation matches `cstep` exactly (difference 3e-15), so I can now look inside it. The full-state comparison with fwd_2 shows a maximum absolute difference of 0.11. That is probably float32-vs-float64 drift on a few entries, since earlier runs found median relative error 1e-7, but I'll check the median alongside.

Now the main measurement, over about 30 cached stays: at insulin bundles, how much each prelude head's slot queries attend to the `u` token and the `z` token, and how stable that is. I'll also measure how much the final state changes when only the insulin rate and dose at that bundle are doubled. That shows how much of the dose actually gets through one core step today.

```python
import time
def ins_atoms_scaled(ats, f):
    o=[]
    for a in ats:
        a2=dict(a)
        if a['code']=='insulin' and a['kind']==0:
            a2['rate']=a['rate']*f; a2['dose']=a['dose']*f
        o.append(a2)
    return o
t0_=time.time()
relerr=[]; A_u=[]; A_z=[]; A_e=[]; dY=[]; dOut=[]; ynorm=[]
for sid in list(bundles_B.keys())[:30]:
    R = run_tf(sid)
    rows = np.where(st_sid==sid)[0]
    Xs = np.stack([np.concatenate([r['out'].numpy(), r['y'].numpy().reshape(-1), r['z'].numpy()]) for r in R])
    relerr.append(np.median(np.abs(Xs-X_[rows]))/np.median(np.abs(X_[rows])))
    Mv=[r['e'] for r in R]; Mt=[r['t'] for r in R]
    for k,r in enumerate(R):
        if not any(a['code']=='insulin' and a['kind']==0 for a in r['ats']): continue
        E, att, x, tok = prelude_full(r['hf'], r['e'], r['u'], r['z'])
        A_e.append(att[:,:,4].mean(1).numpy()); A_u.append(att[:,:,5].mean(1).numpy()); A_z.append(att[:,:,6].mean(1).numpy())
        e2,u2 = enc2(ins_atoms_scaled(r['ats'],2.0))
        lo=max(0,k-127); Mv_=torch.stack(Mv[lo:k]+[e2]); Mage=torch.tensor([(r['t']-tt)*3600.0 for tt in Mt[lo:k+1]],dtype=torch.float64)
        o2,y2,z2,E2,hf2 = cstep(r['out_p'], r['h_p'], r['dt'], e2, u2, Mv_, Mage)
        dY.append(float((y2-r['y']).norm())); dOut.append(float((o2-r['out']).norm())); ynorm.append(float(r['y'].norm()))
print("time", round(time.time()-t0_,1), "s; median rel err per stay:", np.round(np.median(relerr),7), "max", np.round(max(relerr),6))
A_u=np.array(A_u); A_z=np.array(A_z); A_e=np.array(A_e)
print("n insulin bundles:", len(A_u))
print("att to u  per head mean:", A_u.mean(0).round(3), " sd:", A_u.std(0).round(3))
print("att to z  per head mean:", A_z.mean(0).round(3), " sd:", A_z.std(0).round(3))
print("att to e  per head mean:", A_e.mean(0).round(3))
print("|dy| (x2 dose) median:", np.median(dY).round(4), " |y| median:", np.median(ynorm).round(2), " |dout| median:", np.median(dOut).round(4))
```