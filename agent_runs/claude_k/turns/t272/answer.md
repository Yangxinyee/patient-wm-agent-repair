The last run failed inside `u_rec`: it takes scalar rates, and I had passed arrays. Nothing else was wrong. I'm re-running the same attention measurement with scalar inputs. The question is how much attention the four state slots give the dose token (u_k) and the latent token (z) at drug bundles. If some head gives z ≥ 0.1, I can strengthen the dose effect through a dedicated, amplified z channel. If attention to z and u is near zero in every head, that route is closed, and I'll go to a readout-side correction instead.

```python
miss=0; att=[]; zs=[]
for k in ks:
    key=(int(sid[k]),round(float(hr[k]),3))
    ri=rmap.get(key+('insulin',),None); rn=rmap.get(key+('norepinephrine',),None)
    if ri is None and rn is None: miss+=1; continue
    u=np.asarray(u_rec(float(ri or 0.0),float(rn or 0.0))).reshape(-1)[:192]
    hp=xo[k-1,192:960].reshape(4,192)*np.exp(-(hr[k]-hr[k-1])*3600*np.exp(fa))
    zt=Zw@xo[k,960:992]+Zb
    X=np.vstack([hp+SL, ee[None], u[None], zt[None]]); Xn=LNnp(X,n1w,n1b)
    Qm=Xn[:4]@Wq.T; K=(Xn@Wkv.T)[:,:192]
    a=[]
    for h in range(4):
        s=Qm[:,h*48:(h+1)*48]@K[:,h*48:(h+1)*48].T/np.sqrt(48); s=np.exp(s-s.max(1,keepdims=True)); a.append(s/s.sum(1,keepdims=True))
    att.append(np.array(a)); zs.append([zt.std(), np.linalg.norm(zt-zt.mean())])
att=np.array(att); print('n',len(att),'miss',miss)
print('mean att to [s0..s3,e,u,z] per head:\n', np.round(att.mean((0,2)),3))
print('z token sd %.3f, centred norm %.3f'%tuple(np.mean(zs,0)))
```