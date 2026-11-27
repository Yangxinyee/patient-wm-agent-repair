`enc_e_var` returns both encodings for a drug bundle: e from its END atoms and u from its START atoms. The END-atom e is about 18 away from `e_empty()`, so using `e_empty()` would have distorted the reconstruction. I'll now rebuild the prelude at every bundle of the 99 stays (original states). Reading bundles get e from `e_read` and an empty u. Drug bundles get e and u from `enc_e_var`. Then I compute the slot outputs after attention, the FFN input LN, all 768 FFN pre-activations and E. Next I'll use these to find unused FFN units and to fit the gate and dose decoders. A sanity check: drug bundles should show a clearly different FFN-input pattern from reading bundles, which is needed for a clean gate.

```python
cov=sum(1 for (s,h) in SB.keys() if s in set(int(x) for x in sel)); print('SB keys in sel', cov)
Wo=A(load('core.prelude.0.attn.out.weight')); n3w=A(load('core.prelude.0.norm3.weight')); n3b=A(load('core.prelude.0.norm3.bias'))
F0w=A(load('core.prelude.0.ffn.0.weight')); F0b=A(load('core.prelude.0.ffn.0.bias')); F2w=A(load('core.prelude.0.ffn.2.weight')); F2b=A(load('core.prelude.0.ffn.2.bias'))
pnw=A(load('core.prelude_norm.weight')); pnb=A(load('core.prelude_norm.bias'))
def prelude(hp, e, u, zt, ret_att=False):
    X=np.vstack([hp+SL, e[None], u[None], zt[None]]); Xn=LNnp(X,n1w,n1b)
    Qm=Xn[:4]@Wq.T; KV=Xn@Wkv.T; K=KV[:,:192]; V=KV[:,192:]; O=np.zeros((4,192))
    for h in range(4):
        s=Qm[:,h*48:(h+1)*48]@K[:,h*48:(h+1)*48].T/np.sqrt(48); s=np.exp(s-s.max(1,keepdims=True)); s/=s.sum(1,keepdims=True)
        O[:,h*48:(h+1)*48]=s@V[:,h*48:(h+1)*48]
    x1=X[:4]+O@Wo.T; x1n=LNnp(x1,n3w,n3b); pre=x1n@F0w.T+F0b
    x2=x1+gelu_np(pre)@F2w.T+F2b; return x1n, pre, LNnp(x2,pnw,pnb)
typ=np.full(len(hr),-1); FEAT=np.zeros((len(hr),4,192),np.float32); PREF=np.zeros((len(hr),4,768),np.float32); EE=np.zeros((len(hr),4,192),np.float32)
for k in range(1,len(hr)):
    if sid[k]!=sid[k-1]: continue
    key=(int(sid[k]),round(float(hr[k]),3))
    if drug[k]: e,u=[np.asarray(x,float) for x in enc_e_var(key,'none')]; t_=1
    elif k in RB: e=np.asarray(e_read(k),float); u=np.asarray(U_EMPTY,float).reshape(-1)[:192]; t_=0
    else: continue
    hp=xo[k-1,192:960].reshape(4,192)*np.exp(-(hr[k]-hr[k-1])*3600*np.exp(fa))
    a_,b_,c_=prelude(hp,e,u,Zw@xo[k,960:992]+Zb); FEAT[k],PREF[k],EE[k]=a_,b_,c_; typ[k]=t_
print('reading',(typ==0).sum(),'drug',(typ==1).sum())
mx=PREF[typ>=0].max((0,1)); print('units max pre < -3:', np.where(mx<-3)[0][:20], 'n', (mx<-3).sum(), 'lowest', np.round(np.sort(mx)[:6],2))
```