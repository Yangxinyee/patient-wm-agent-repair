The register is solved exactly (error 2e-5). `last_numeric` input column 0 is has_last, column 3 is log(1+age in s)/10 with age measured to the predicted bundle, and column 6 is the z-score of the last value. There is no extra flag and no status term. Together with the exact core step, I now have an exact replica of the whole factual pathway.

Next I'll record this and write a batched step (many paths × dose conditions at once) for rollouts. I'll first check that the batched step reproduces the recorded states (fwd_2) under teacher forcing on one stay. A match at ~1e-5 confirms the batched implementation. I'm saving the code to a file so it survives a kernel reset.

```python
with open('/work/notes.md','a') as f:
    f.write("""
- HEAD EXACT (run 166, fwd_5 probe: z_head.weight=0 -> deterministic; value NLL 1.55632 vs orig 1.55781):
  view = out + out_norm(mean hf) + gap(log dt/10) + z_head(z~prior); q=query_norm(view+type_emb[1]);
  with_code = q + code_emb[c] + last_numeric(f) + bins.value[last vb] + bins.rate[0]; f[0]=has_last, f[3]=log1p(age_s to predicted bundle)/10, f[6]=z_last (no status).
  logits = value_bin mlp(gelu) + embed_transition.value_bin[c*17+last vb]; mean=sum p[1:]*centres[code]. ALL acts gelu (ACTMAP['']='gelu').
""")
code_wm2 = r'''
import torch, math, numpy as np
import torch.nn.functional as F
T=torch.float64
G_=F.gelu
def blin(W,x,n,bias=True):
    y=x@W[n+'.weight'].T
    if bias and (n+'.bias') in W: y=y+W[n+'.bias']
    return y
def bmlp(W,x,n): return blin(W,G_(blin(W,x,n+'.0')),n+'.2')
def bln(W,x,n): return F.layer_norm(x,(x.shape[-1],),W[n+'.weight'],W[n+'.bias'],1e-5)
def bmha(W,xq,xkv,n,H=4):
    # xq (B,nq,192) xkv (B,nk,192)
    B,nq,_=xq.shape; nk=xkv.shape[1]; dh=192//H
    q=(xq@W[n+'.q.weight'].T).reshape(B,nq,H,dh).transpose(1,2)
    kv=xkv@W[n+'.kv.weight'].T; k=kv[...,:192].reshape(B,nk,H,dh).transpose(1,2); v=kv[...,192:].reshape(B,nk,H,dh).transpose(1,2)
    att=torch.softmax((q@k.transpose(-1,-2))/math.sqrt(dh),-1)
    o=(att@v).transpose(1,2).reshape(B,nq,192)
    return o@W[n+'.out.weight'].T
def bself(W,x,L):
    a=bln(W,x,L+'.norm1'); x=x+bmha(W,a,a,L+'.attn')
    return x+bmlp(W,bln(W,x,L+'.norm3'),L+'.ffn')
def bstep(W,out,h,dt,e,u,Mt,ages):
    # out (B,192) h (B,4,192) e,u (B,192) Mt (B,n,192) ages (n,) seconds
    hf=h*torch.exp(-dt*torch.exp(W['core.flow.a']))
    g=bmlp(W,torch.tensor([[math.log(dt)/10]],dtype=T),'gap')
    ctx=out+bln(W,hf.mean(1),'core.out_norm')+g
    z=bmlp(W,torch.cat([ctx,e,u],-1),'z_post')[:,:32]
    tok=torch.cat([hf+W['core.slots'],e[:,None],u[:,None],blin(W,z,'z_in')[:,None]],1)
    tok=bself(W,tok,'core.prelude.0'); E=bln(W,tok[:,:4],'core.prelude_norm')
    ae=bmlp(W,torch.log1p(ages)[:,None],'core.age')
    Mk=Mt+ae[None]
    x=hf+E
    for L in ['core.blocks.0.0','core.blocks.0.1']:
        a_=bln(W,x,L+'.norm1'); x=x+bmha(W,a_,a_,L+'.attn')
        x=x+bmha(W,bln(W,x,L+'.norm2'),Mk,L+'.cross')
        x=x+bmlp(W,bln(W,x,L+'.norm3'),L+'.ffn')
    sp=F.softplus(W['core.delta_in']); abar=torch.exp(-sp*torch.exp(W['core.a_in']))
    h2=abar*hf+sp*(E@W['core.b_in.weight'].T)+(x-(hf+E))
    y=bself(W,h2@W['core.c_out.weight'].T,'core.coda.0')
    return bln(W,y.mean(1),'core.out_norm'), y, z
def bview(W,out,h,dt,zs=None,gen=None,nz=4):
    # returns views (B,nz,192) ; zs: if None draw from prior
    hf=h*torch.exp(-dt*torch.exp(W['core.flow.a']))
    base=out+bln(W,hf.mean(1),'core.out_norm')+bmlp(W,torch.tensor([[math.log(dt)/10]],dtype=T),'gap')
    zp=bmlp(W,base,'z_prior')
    if zs is None:
        eps=torch.randn(base.shape[0],nz,32,generator=gen,dtype=T)
    else: eps=zs
    z=zp[:,None,:32]+torch.exp(zp[:,None,32:])*eps
    return base[:,None,:]+blin(W,z,'z_head')
def bvalue_probs(W,views,code,has,age_s,zlast,vblast):
    # views (B,nz,192); has,age_s,zlast (B,), vblast (B,) long
    q=bln(W,views+W['type_emb.weight'][1],'query_norm')
    B=views.shape[0]; f=torch.zeros(B,8,dtype=T); f[:,0]=has; f[:,3]=torch.log1p(age_s)/10*has; f[:,6]=zlast
    wc=q+(W['embed.code.weight'][code]+blin(W,f,'last_numeric')+W['embed.bins.value.weight'][vblast]+W['embed.bins.rate.weight'][0])[:,None,:]
    lg=bmlp(W,wc,'attr_heads.value_bin')+W['embed_transition.value_bin.weight'][code*17+vblast][:,None,:]
    return torch.softmax(lg,-1).mean(1)   # (B,17)
'''
open('/work/wm2.py','w').write(code_wm2); exec(code_wm2)
# verify bstep vs Xs on first stay (teacher forced, B=1)
s=ks[0]; i0=row_start[s]; B_=BUN[s]; Ms=[]; ts=[]; errs=[]
for k in range(len(B_)):
    e,u=enc_eu2(B_[k][1])
    if k>0:
        prev=torch.tensor(Xs[i0+k-1],dtype=T)
        Mt=torch.stack(Ms+[e])[None]; ages=torch.tensor([(B_[k][0]-tj)*3600.0 for tj in ts+[B_[k][0]]],dtype=T)
        o2,y2,z2=bstep(W0,prev[None,:192],prev[192:960].reshape(1,4,192),(B_[k][0]-B_[k-1][0])*3600.0,e[None],u[None],Mt,ages)
        errs.append(float((y2.reshape(-1)-torch.tensor(Xs[i0+k,192:960])).abs().max()))
    Ms.append(e); ts.append(B_[k][0])
print('bstep max err median', np.median(errs), 'max', max(errs))
print('u_empty zero?', float(u_empty.abs().max()))
```