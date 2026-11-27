The disassembly shows the cause. In my verified z computation, a bundle with no measurement atoms gets e = **zero vector**, not the empty-set encoding, and a bundle with no actions gets u = zero. My new step used the empty-set encoding for absent e, so every action bundle was wrong.

Next I rerun the one-step variant test with absent e set to zero. In the memory, I either skip action-only bundles or include their zero vectors. If `z` now matches to about 1e-5, I can judge the core variants by their `out`/`h` error. One combination giving near-zero error everywhere would confirm the exact core.

```python
def onestep_err2(sid, opt, inc_cur, skip_empty, nmax=60, k0=1):
    bl = BUN[sid]; rows = row_of[sid]; times=[t for t,_ in bl]
    enc=[]
    for t,atoms in bl:
        meas=[a for a in atoms if not a['act']]; act=[a for a in atoms if a['act']]
        e = set_enc(W0, meas, 'encode') if meas else torch.zeros(192,dtype=T)
        u = set_enc(W0, act, 'encode_action') if act else torch.zeros(192,dtype=T)
        enc.append((e,u,len(meas)>0))
    errs=[]
    for k in range(k0, min(len(bl),nmax)):
        prev = torch.tensor(X[rows[k-1]],dtype=T); out=prev[:192]; h=prev[192:960].reshape(4,192)
        dt=(times[k]-times[k-1])*3600
        js=[j for j in range(k+(1 if inc_cur else 0)) if (enc[j][2] or not skip_empty)][-128:]
        Mt=torch.stack([enc[j][0] for j in js]) if js else None
        ages=torch.tensor([(times[k]-times[j])*3600 for j in js],dtype=T) if js else None
        e,u,_=enc[k]
        o2,y,z,_=core_step(W0,out,h,dt,e,u,Mt=Mt,ages=ages,opt=opt)
        tgt=torch.tensor(X[rows[k]],dtype=T)
        errs.append([(o2-tgt[:192]).abs().max().item(),(y.reshape(-1)-tgt[192:960]).abs().max().item(),(z-tgt[960:]).abs().max().item()])
    return np.array(errs)
for inc in [False,True]:
    for skip in [False,True]:
        for au in ['s','h']:
            for am in ['key','input']:
                er=onestep_err2(2020000000,dict(age_unit=au,age_mode=am),inc,skip,nmax=30)
                print(inc,skip,au,am,'out med/max %.3g %.3g | h med/max %.3g %.3g | z max %.3g'%(np.median(er[:,0]),er[:,0].max(),np.median(er[:,1]),er[:,1].max(),er[:,2].max()))
```