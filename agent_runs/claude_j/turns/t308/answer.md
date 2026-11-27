The probe works well within the training range.
- **Drug I:** the rate is read back almost exactly at x0–x2 (3.38 against a true 3.39 at x2). It under-reads slightly at x3 (4.73 against 5.08).
- **Drug N:** it saturates above the training cap (0.91 against 1.00 at x2, 1.12 against 1.50 at x3).

So the action token already carries a usable, mostly linear rate signal. I may later add a numeric pathway for drug N extrapolation.

Next I'll measure how the prelude routes this token into the slots. From the GPU states I'll rebuild the prelude tokens at each drug bundle and compute, for each of the 4 heads, the attention weight from each slot to the u token, with its mean and spread. I'll also keep the normalised vectors of the other tokens, so the injection can be made blind to them.
- **Confirms the route:** a head with mean attention to u of at least 0.2 and coefficient of variation below about 0.5.
- **Refutes it:** all attention weights are tiny or erratic.

```python
Wq=T64(W[P+'.attn.q.weight']); Wkv=T64(W[P+'.attn.kv.weight'])
fa=T64(W['core.flow.a']); SL=T64(W['core.slots'])
AL=[]; OTH=[]; UL=[]; LU=[]
for s in stays:
    X=T64(X_of(s)); B=BUN[s]
    for k in range(1,len(B)):
        mn,evs=B[k]
        if not any(a[0] in ('insulin','norepinephrine') and a[1]=='start' for a in evs): continue
        dt=(B[k][0]-B[k-1][0])*60.0
        hp=X[k-1,192:960].reshape(4,192)*torch.exp(-dt*torch.exp(fa))
        e,u=bvd(evs); z=X[k,960:992]; zt=T64(linp(z,'z_in'))
        tok=torch.cat([hp+SL, T64(e)[None], T64(u)[None], zt.reshape(1,192)],0)
        x=ln1(tok); q=(x[:4]@Wq.T).reshape(4,4,48); kk=(x@Wkv.T)[:,:192].reshape(7,4,48)
        a=torch.softmax(torch.einsum('ihd,jhd->hij',q,kk)/np.sqrt(48),-1)
        AL.append(a.numpy()); OTH.append(x[[0,1,2,3,4,6]].numpy()); LU.append(x[5].numpy()); UL.append((s,k))
AL=np.stack(AL); print(len(AL))
for H in range(4):
    au=AL[:,H,:,5]; print(H,'u',np.round(au.mean(0),3),'cv',np.round(au.std(0)/au.mean(0),2),'tok-mean',np.round(AL[:,H].mean((0,1)),2))
```