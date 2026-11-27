Everything needed on the z_post side is already in place and exact:
- Unit 12's pre-activation is 20.0 at every drug bundle and ≤ −20 elsewhere, a perfect gate.
- Unit 88's is 20 + insulin innovation (correlation 0.996, slope 0.99).
- Unit 105's is 20 + 10 × NE innovation (correlation 0.989).
- All three are off at non-drug bundles.
- The four END-atom e vectors are exactly constant within each pattern.

So the z posterior can carry, at drug bundles only, a pattern one-hot plus the two innovations. The remaining step is to design what the z token becomes.

Per pattern, I'll start from an exact copy of e's LayerNorm image, then add a correction Δn (closed-form minimum-norm least squares) that:
- keeps head 1's value exactly equal to e's;
- raises z's head-1 logit well above e's, so z takes over head 1;
- pushes z's logits in heads 0, 2 and 3 well below the best competing token.

I'll then check exact margins over every drug bundle and slot. Success means a head-1 margin ≥ +8 and margins in the other heads ≤ −8 everywhere, with a small value error.

```python
Wk=Wkv[:192].astype(float); Wv=Wkv[192:].astype(float); g1=n1w.astype(float); be1=n1b.astype(float)
def nrm(x): x=x-x.mean(); return x/np.sqrt((x**2).mean()+1e-5)
Av=Wv[48:96]*g1[None,:]
DES={}
for p in range(4):
    b=np.where(pid==p)[0]; ne_=nrm(EP[p]); te=LNnp(EP[p][None],n1w,n1b)[0]
    cbar=[]; 
    for h in range(4):
        Qh=Qs[b][:,:,h*48:(h+1)*48].reshape(-1,48); cbar.append(((Qh@Wk[h*48:(h+1)*48])*g1[None,:]).mean(0)/np.sqrt(48))
    tg=[-12.,18.,-25.,-25.]; r=1.0
    for it in range(4):
        C=np.vstack(cbar+[Av, np.ones(192), ne_]); d=np.r_[tg, (r-1)*(Av@ne_), 0., 0.]
        dn=C.T@np.linalg.lstsq(C@C.T, d, rcond=None)[0]; r=np.sqrt(1+(dn**2).mean())
    t=LNnp((ne_+dn)[None],n1w,n1b)[0]
    mg=[]
    for h in range(4):
        Qh=Qs[b][:,:,h*48:(h+1)*48]; zl=(Qh@(Wk[h*48:(h+1)*48]@t))/np.sqrt(48); mo=Ls[b][:,h,:,:6].max(-1); mg.append(zl-mo)
    DES[p]=(ne_+dn)
    print('p',p,'n',len(b),'|dn| %.2f'%np.linalg.norm(dn),'h1 min margin %.1f'%mg[1].min(),'h0/h2/h3 max margin %.1f %.1f %.1f'%(mg[0].max(),mg[2].max(),mg[3].max()),'value err %.4f (|V| %.2f)'%(np.linalg.norm(Wv[48:96]@t-Wv[48:96]@te),np.linalg.norm(Wv[48:96]@te)))
```