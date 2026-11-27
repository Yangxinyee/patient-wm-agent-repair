Very informative. In the prelude FFN input, the context-expected rate is almost perfectly decodable (R² ≈ 0.99), but the dose itself is not: R² is about 0.6 to 0.75, whatever the regularisation or slot. The dose reaches the slots only through the u-token value path, and that compresses it, so no FFN unit can compute the innovation cleanly. The only place where the innovation is computed accurately is z_post, which reads u directly. Its output reaches the state only through the z token, which every head currently ignores.

That points to a more promising design. If the z token is also ignored at reading bundles, z_post and z_in are nearly free to repurpose. At drug bundles I could set the z token to a designed vector that takes over head 1, which currently attends 100% to the END-atom e token, while staying ignored by heads 0, 2 and 3. Its head-1 value would reproduce e's value plus the innovation times a chosen direction.

To check this I need four things:
1. Attention to z at reading bundles. Near zero confirms z is free to repurpose.
2. Per-head query spreads and logit gaps at drug bundles, to show whether one key can dominate head 1 while losing in the other heads.
3. How many distinct END-atom e patterns occur.
4. How much head 1's e-value varies across those patterns.

```python
def attn_all(hp,e,u,zt):
    X=np.vstack([hp+SL, e[None], u[None], zt[None]]); Xn=LNnp(X,n1w,n1b)
    Qm=Xn[:4]@Wq.T; KV=Xn@Wkv.T; K=KV[:,:192]; V=KV[:,192:]; out=[]
    for h in range(4):
        s=Qm[:,h*48:(h+1)*48]@K[:,h*48:(h+1)*48].T/np.sqrt(48); out.append(s)
    return np.array(out), Qm, V   # logits (4 heads,4 slots,7 tokens)
rz=[]; 
for k in np.where(typ==0)[0][::3]:
    hp=xo[k-1,192:960].reshape(4,192)*np.exp(-(hr[k]-hr[k-1])*3600*np.exp(fa))
    lg,_,_=attn_all(hp,np.asarray(e_read(k),float),np.asarray(U_EMPTY,float).reshape(-1)[:192],Zw@xo[k,960:992]+Zb)
    p=np.exp(lg-lg.max(2,keepdims=True)); p/=p.sum(2,keepdims=True); rz.append(p.mean(1))
print('reading-bundle mean att [s0..s3,e,u,z] per head:\n',np.round(np.mean(rz,0),3))
Qs=[];Ls=[];Ve=[];pats=[]
for k in kd:
    key=(int(sid[k]),round(float(hr[k]),3)); e,u=[np.asarray(x,float) for x in enc_e_var(key,'none')]
    hp=xo[k-1,192:960].reshape(4,192)*np.exp(-(hr[k]-hr[k-1])*3600*np.exp(fa))
    lg,Qm,V=attn_all(hp,e,u,Zw@xo[k,960:992]+Zb); Qs.append(Qm); Ls.append(lg); Ve.append(V[4,48:96]); pats.append(tuple(np.round(e[:3],3)))
Qs=np.array(Qs); Ls=np.array(Ls); Ve=np.array(Ve)
gap=Ls[:,:,:,6]-np.delete(Ls,6,axis=3).max(3)
print('z logit minus best other, per head (mean, max):', [(round(float(gap[:,h].mean()),1),round(float(gap[:,h].max()),1)) for h in range(4)])
import collections; pc=collections.Counter(pats); print('END patterns', len(pc), sorted(pc.values(),reverse=True)[:6])
print('head1 V_e spread: mean norm %.3f, sd across bundles %.3f'%(np.linalg.norm(Ve.mean(0)), np.linalg.norm(Ve.std(0))))
```