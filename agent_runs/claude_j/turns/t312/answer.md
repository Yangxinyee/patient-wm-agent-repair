The source files show the calling pattern:
- `heads_full(O, H, rt, eps)` gives per-reading logits;
- `mean_of(lg, ev)` turns them into predicted means;
- `eps` is a seeded set of 4 prior draws per reading.

Now the main design experiment. I'll inject a small rate signal into head 2's value along each of its 48 value dimensions, one at a time, on 25 tuning stays in the replica, using the drug-I probe. For each direction I record:
- (a) the change in core features at readings, measured against the factual feature covariance;
- (b) the change in all predicted reading means, which is the disturbance.

Solving a generalized eigenproblem then gives the value direction p with the best signal-to-disturbance ratio. I'll also check how the injected signal persists with time since the last drug bundle.
- **Confirms the design:** a direction with a large signal and small disturbance (RMS shift of predicted means well under 1 per unit of signal) that persists for a few hours.
- **Refutes it:** every direction disturbs the predictions as much as it signals.

```python
import time
from scipy.linalg import eigh
t0=time.time(); KV=P+'.attn.kv.weight'; Wkv0=W[KV].clone(); dty=Wkv0.dtype
wI=torch.as_tensor(PRB['insulin'],dtype=dty); ST=stays[:25]; kap=0.1
def feats(O,H,rt):
    pr=rt['prow'].values.astype(int); return torch.cat([T64(O)[pr],T64(H)[pr].mean(1)],1).numpy()
info={}
for s in ST:
    rt=RTs[s]; B=BUN[s]; sig=np.zeros(len(B)); mins=np.array([b[0] for b in B],float)
    for k,(mn,evs) in enumerate(B):
        if any(a[0]=='insulin' and a[1]=='start' for a in evs): sig[k]=float(wI@ln1(bvd(evs)[1]).to(dty))
    eps_=torch.randn(len(rt),4,32,generator=torch.Generator().manual_seed(int(s)%100000),dtype=torch.float64)
    O,H,_=run_cf(s,{},BUN[s][1][0],X_of(s)); lg=heads_full(O,H,rt,eps_); ev=rt['event'].values; pr=rt['prow'].values.astype(int)
    li=np.array([mins[np.where(sig[:p+1]>0)[0][-1]] if (sig[:p+1]>0).any() else -1e9 for p in pr])
    info[s]=dict(sig=sig,eps=eps_,f=feats(O,H,rt),pm=mean_of(lg,ev),pr=pr,ev=ev,age=(rt['minute'].values-li)/60.)
print('base', round(time.time()-t0,1))
DF=[]; DP=[]
try:
    for i in range(48):
        Wn=Wkv0.clone(); Wn[288+i]+=kap*wI; W[KV]=Wn; dfi=[]; dpi=[]
        for s in ST:
            I=info[s]; O,H,_=run_cf(s,{},BUN[s][1][0],X_of(s)); lg=heads_full(O,H,RTs[s],I['eps'])
            dfi.append(feats(O,H,RTs[s])-I['f']); dpi.append(mean_of(lg,I['ev'])-I['pm'])
        DF.append(np.concatenate(dfi).astype(np.float32)); DP.append(np.concatenate(dpi))
finally: W[KV]=Wkv0
DF=np.stack(DF,-1)/kap; DP=np.stack(DP,-1)/kap
F0=np.concatenate([info[s]['f'] for s in ST]); SG=np.concatenate([info[s]['sig'][info[s]['pr']] for s in ST]); AGE=np.concatenate([info[s]['age'] for s in ST])
imm=SG>0.3; C0=np.cov(F0.T); Sf=C0+1e-3*np.trace(C0)/384*np.eye(384); Si=np.linalg.inv(Sf)
J=(DF[imm]/SG[imm,None,None]).mean(0); G=J.T@Si@J; Q=DP.T@DP/len(DP)
lam,V=eigh(G,Q+1e-4*np.trace(Q)/48*np.eye(48)); p=V[:,-1]
print('time',round(time.time()-t0,1),'n',len(SG),'imm',imm.sum(),'lam',np.round(lam[-3:],1))
print('snr/unit',round(float(np.sqrt(p@G@p)),3),'dist',round(float(np.sqrt(p@Q@p)),4), 'raw best-dir snr', round(float(np.sqrt(np.max(np.diag(G)))),3))
dfp=DF@p; m=np.sqrt(np.einsum('ni,ij,nj->n',dfp,Si,dfp))
for lo,hi in ((0,1.01),(1.01,2.01),(2.01,3.01),(3.01,6),(6,99)):
    i=(AGE>lo)&(AGE<=hi); print(lo,hi,i.sum(), round(float(np.median(m[i])) if i.any() else 0,3))
```