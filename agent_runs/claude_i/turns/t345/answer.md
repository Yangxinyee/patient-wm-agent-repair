The prelude has bias-free query, key/value and output projections, plus pre-norm layer norms (`norm1` before attention, `norm3` before the FFN). That makes the rank-1 value-projection idea concrete, but it is only clean if two things hold:
1. There is a direction in normalised-token space along which slot, e, u and z tokens hardly vary. That bounds the side effects of a large gain.
2. Slots pay non-negligible attention to the z token, which sets how much gain is needed.

Next I'll compute both from the stored states:
- **Tokens:** rebuild all 7 prelude tokens per bundle (slots from the flowed previous state, plus e, u and the z token).
- **Quiet directions:** take the smallest eigenvalues of their normalised second-moment matrix, constrained so the direction can't pick up the layer-norm bias.
- **Attention:** the per-head attention slots give to the e, u and z tokens at insulin and NE bundles. I'll also read the number of heads from the config.

If the smallest eigenvalues are tiny (≲1e-5) and z-token attention is not negligible, the approach is viable. If eigenvalues are ~1e-3 or attention ~0, it isn't, and I'll stop editing and keep the original model.

```python
cfg=VOCAB['config']; print({k:v for k,v in cfg.items() if any(s in str(k) for s in ('head','dim','slot','layer'))} if isinstance(cfg,dict) else str(cfg)[:300])
X3a=np.asarray(X3,dtype=np.float64); Hst=X3a[:,192:960].reshape(-1,4,192)
fa_=np.asarray(load_original('core.flow.a'),dtype=np.float64); slots_=np.asarray(load_original('core.slots'),dtype=np.float64)
dtsA=np.asarray(dts,dtype=np.float64); first=np.r_[True,S3a[1:]!=S3a[:-1]]
Hprev=np.where(first[:,None,None],0.0,np.roll(Hst,1,axis=0)); HF=Hprev*np.exp(-dtsA[:,None]*np.exp(fa_)[None,:])[:,None,:]
Zt=np.asarray(zp,dtype=np.float64)[:,:32]@Wzo.T+bzo
TOK=np.concatenate([HF+slots_[None],np.asarray(x,dtype=np.float64)[:,None,192:384],np.asarray(x,dtype=np.float64)[:,None,384:],Zt[:,None,:]],1)
g1=np.asarray(load_original('core.prelude.0.norm1.weight'),dtype=np.float64); be1=np.asarray(load_original('core.prelude.0.norm1.bias'),dtype=np.float64)
mu_=TOK.mean(-1,keepdims=True); sdv=np.sqrt(TOK.var(-1,keepdims=True)+1e-5); XH=(TOK-mu_)/sdv; LN1=g1*XH+be1
keep=np.abs(g1)>0.05*np.median(np.abs(g1)); print('kept dims',keep.sum(),'|g1| min/med',np.abs(g1).min().round(4),np.median(np.abs(g1)).round(3))
Xr=XH[okA].reshape(-1,192)[:,keep]; S_=Xr.T@Xr/len(Xr)
C_=np.c_[np.ones(keep.sum()),(be1/g1)[keep]]; Qc,_=np.linalg.qr(C_); Pc=np.eye(keep.sum())-Qc@Qc.T
ev,EV=np.linalg.eigh(Pc@S_@Pc+1e3*Qc@Qc.T); print('smallest eig',ev[:6])
for nm_,tt in [('slots',XH[okA][:,:4]),('e',XH[okA][:,4]),('u',XH[okA][:,5]),('z',XH[okA][:,6])]:
    pr=(tt.reshape(-1,192)[:,keep]@EV[:,:3]); print(nm_,'sd on 3 quietest',pr.std(0).round(5))
nh=int(cfg.get('n_heads',cfg.get('heads',4))) if isinstance(cfg,dict) else 4
Wq=np.asarray(load_original('core.prelude.0.attn.q.weight'),dtype=np.float64); Wkv=np.asarray(load_original('core.prelude.0.attn.kv.weight'),dtype=np.float64); Wk,Wv=Wkv[:192],Wkv[192:]
dh=192//nh; Q_=(LN1[:,:4]@Wq.T).reshape(-1,4,nh,dh); K_=(LN1@Wk.T).reshape(-1,7,nh,dh)
sc=np.einsum('nshd,nthd->nhst',Q_,K_)/np.sqrt(dh); sc-=sc.max(-1,keepdims=True); AT=np.exp(sc); AT/=AT.sum(-1,keepdims=True)
for nm_,m_ in [('ins',hi_&okA),('ne',hn_&okA)]:
    print(nm_,'nh',nh,'mean attn slot->[e,u,z] per head:',AT[m_][:,:,:,4:].mean((0,2)).round(3).tolist())
```