Run 280 captured 798 drug bundles, 447 with insulin and 599 with norepinephrine, from 40 stays. The prelude layer has bias-free `q` (192×192), `kv` (384×192) and `out` projections, 4 heads of 48 dims, and LayerNorm `norm1` before attention. That is enough to compute the attention logits myself.

Next steps:
1. Check that the computation reproduces the known hard pattern: heads 0–2 put almost all weight on u at drug bundles; head 3 attends the slots.
2. Compute the gradient of every slot/head logit for the z token with respect to that token, at z = copy of u.
3. Find the subspace these gradients span, and test how much of my old added-signal direction (`VD_`) lies in it.

If the old direction has a large logit gradient, that confirms why gains above 1 were neutralised. Projecting it onto the logit-neutral complement should then give a direction whose effect scales with gain. If the gradients turn out to fill all 192 dimensions, this route is closed and I need another one.

```python
nW=lambda k: W_cur[k].double()
g1,b1_=nW('core.prelude.0.norm1.weight'),nW('core.prelude.0.norm1.bias')
Wq=nW('core.prelude.0.attn.q.weight'); Wkv=nW('core.prelude.0.attn.kv.weight'); Wk_,Wv_=Wkv[:192],Wkv[192:]
SL=nW('core.slots')
def ln(x): mu=x.mean(-1,keepdim=True); sd=torch.sqrt(x.var(-1,unbiased=False,keepdim=True)+1e-5); return (x-mu)/sd, sd
def keygrads(c):
    tok=torch.cat([c['hf'].double()+SL, c['e'].double()[None], c['u'].double()[None]],0)  # 6 tokens
    xh,sd=ln(tok); a=xh*g1+b1_
    Q=(a[:4]@Wq.T).reshape(4,4,48); K=(a@Wk_.T).reshape(6,4,48)
    lg=torch.einsum('shd,jhd->shj',Q,K)/math.sqrt(48); w=torch.softmax(lg,-1)
    xu=xh[5]; G=[]
    for s_ in range(4):
        for h_ in range(4):
            cvec=g1*(Wk_[48*h_:48*h_+48].T@Q[s_,h_])/math.sqrt(48)
            gg=(cvec-cvec.mean()-xu*(xu@cvec)/192)/sd[5,0]; G.append(gg)
    return w, torch.stack(G)
Wl=[];GL=[];flag=[]
for c in CAPT:
    w,G=keygrads(c); Wl.append(w); GL.append(G); flag.append((c['info']['hi']>0, c['info']['hn']>0))
Wl=torch.stack(Wl); GL=torch.stack(GL); flag=np.array(flag)
print('mean attn on u (token5) per head, avg over slots:', np.round(Wl[:,:,:,5].mean((0,1)).numpy(),3), '| on slots:', np.round(Wl[:,:,:,:4].sum(-1).mean((0,1)).numpy(),3))
Gm=GL.reshape(-1,192).numpy(); Uq,Sq,Vq=np.linalg.svd(Gm,full_matrices=False); en=np.cumsum(Sq**2)/np.sum(Sq**2)
print('keygrad energy in top k (4,8,16,32,64):', [round(float(en[k-1]),3) for k in [4,8,16,32,64]])
print('CODES4',CODES4,'| VD_ norms', {k:round(float(np.linalg.norm(v)),3) for k,v in VD_.items()})
for dn in ['ins','ne']:
    v=np.asarray(VD_[dn],dtype=float); sens=np.abs(Gm@v).mean(); 
    print(dn,'old dir: mean|logit grad.v|',round(float(sens),3),'| frac of v in top16 keygrad subspace', round(float(np.linalg.norm(Vq[:16]@v)**2/np.dot(v,v)),3), 'top32', round(float(np.linalg.norm(Vq[:32]@v)**2/np.dot(v,v)),3))
```