The run stopped at `mean_of` because `heads_full` returns several objects rather than a single logits tensor, so its output must be unpacked first.

I'll add a small wrapper that picks out the per-reading logits (shape n×4 draws×17 bins), print the structure once, and rerun the same 48-direction design experiment unchanged. The confirm/refute criteria are as before: look for a value direction with high signal-to-disturbance and a persistent signal.

```python
def hf(O,H,rt,eps_):
    r=heads_full(O,H,rt,eps_)
    if isinstance(r,(tuple,list)):
        for x in r:
            if torch.is_tensor(x) and x.dim()==3 and x.shape[-1]==17: return x
    return r
s=ST[0]; r_=heads_full(*run_cf(s,{},BUN[s][1][0],X_of(s))[:2],RTs[s],torch.zeros(len(RTs[s]),4,32,dtype=torch.float64))
print(type(r_).__name__, [ (type(x).__name__, tuple(getattr(x,'shape',()))) for x in (r_ if isinstance(r_,(tuple,list)) else [r_])][:6])
t0=time.time(); info={}
for s in ST:
    rt=RTs[s]; B=BUN[s]; sig=np.zeros(len(B)); mins=np.array([b[0] for b in B],float)
    for k,(mn,evs) in enumerate(B):
        if any(a[0]=='insulin' and a[1]=='start' for a in evs): sig[k]=float(wI@ln1(bvd(evs)[1]).to(dty))
    eps_=torch.randn(len(rt),4,32,generator=torch.Generator().manual_seed(int(s)%100000),dtype=torch.float64)
    O,H,_=run_cf(s,{},BUN[s][1][0],X_of(s)); lg=hf(O,H,rt,eps_); ev=rt['event'].values; pr=rt['prow'].values.astype(int)
    li=np.array([mins[np.where(sig[:p+1]>0)[0][-1]] if (sig[:p+1]>0).any() else -1e9 for p in pr])
    info[s]=dict(sig=sig,eps=eps_,f=feats(O,H,rt),pm=mean_of(lg,ev),pr=pr,ev=ev,age=(rt['minute'].values-li)/60.)
DF=[]; DP=[]
try:
    for i in range(48):
        Wn=Wkv0.clone(); Wn[288+i]+=kap*wI; W[KV]=Wn; dfi=[]; dpi=[]
        for s in ST:
            I=info[s]; O,H,_=run_cf(s,{},BUN[s][1][0],X_of(s)); lg=hf(O,H,RTs[s],I['eps'])
            dfi.append(feats(O,H,RTs[s])-I['f']); dpi.append(mean_of(lg,I['ev'])-I['pm'])
        DF.append(np.concatenate(dfi).astype(np.float32)); DP.append(np.concatenate(dpi))
finally: W[KV]=Wkv0
DF=np.stack(DF,-1)/kap; DP=np.stack(DP,-1)/kap
F0=np.concatenate([info[s]['f'] for s in ST]); SG=np.concatenate([info[s]['sig'][info[s]['pr']] for s in ST]); AGE=np.concatenate([info[s]['age'] for s in ST])
imm=SG>0.3; C0=np.cov(F0.T); Sf=C0+1e-3*np.trace(C0)/384*np.eye(384); Si=np.linalg.inv(Sf)
J=(DF[imm]/SG[imm,None,None]).mean(0); G=J.T@Si@J; Q=DP.T@DP/len(DP)
lam,V=eigh(G,Q+1e-4*np.trace(Q)/48*np.eye(48)); p=V[:,-1]
print('time',round(time.time()-t0,1),'n',len(SG),'imm',imm.sum(),'lam',np.round(lam[-3:],1))
print('snr/unit',round(float(np.sqrt(p@G@p)),3),'dist',round(float(np.sqrt(p@Q@p)),4),'best-diag snr',round(float(np.sqrt(np.max(np.diag(G)))),3))
dfp=DF@p; m=np.sqrt(np.einsum('ni,ij,nj->n',dfp,Si,dfp))
print([ (lo,int(((AGE>lo)&(AGE<=hi)).sum()), round(float(np.median(m[(AGE>lo)&(AGE<=hi)])),3)) for lo,hi in ((0,1.01),(1.01,2.01),(2.01,3.01),(3.01,6),(6,99)) if ((AGE>lo)&(AGE<=hi)).any()])
```