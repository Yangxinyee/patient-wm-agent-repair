All 20 stays here have 54 bundles: three pre-admission bundles, an hourly measurement bundle and a 5-minute drug bundle for 25 hours, and a final END bundle. e and u are never exactly zero; empty sets encode to constant vectors. So the memory M for bundle k is simply the e vectors of bundles 0..k of the same stay, at most 54 entries.

Next I'll add the block step to /work/lib2.py:
- self-attention over slots, cross-attention to M plus the age embedding, and FFN, for both block layers;
- the h update, c_out, coda and out.

I'll validate it against `tf_loop`'s stored hn and its returned h and out, testing both "age added to keys and values" and "keys only". An exact match tells me the cross-attention form and gives exact block internals for fitting the gate.

```python
print('age shapes', SHAPES['core.age.0.weight'], SHAPES['core.age.2.weight'], 'ffn2', SHAPES['core.blocks.0.0.ffn.2.weight'])
code3=r'''
def mh(P, pre, xq, xkv, mask=None, kadd=None):
    B,Tq,_=xq.shape; Tk=xkv.shape[1]; H=4; dh=48
    q=F.linear(xq, P[pre+'.q.weight']); k,v=F.linear(xkv, P[pre+'.kv.weight']).split(192,-1)
    if kadd is not None: k,_=F.linear(xkv+kadd, P[pre+'.kv.weight']).split(192,-1)
    qh=q.view(B,Tq,H,dh).transpose(1,2); kh=k.view(B,Tk,H,dh).transpose(1,2); vh=v.view(B,Tk,H,dh).transpose(1,2)
    s=qh@kh.transpose(-1,-2)/math.sqrt(dh)
    if mask is not None: s=s.masked_fill(~mask[:,None,None,:], float('-inf'))
    a=torch.softmax(s,-1); return F.linear((a@vh).transpose(1,2).reshape(B,Tq,192), P[pre+'.out.weight'])
def ffP(P, pre, x): return F.linear(F.gelu(F.linear(x, P[pre+'.0.weight'], P[pre+'.0.bias'])), P[pre+'.2.weight'], P[pre+'.2.bias'])
def agE(P, age_s):
    a=torch.log1p(age_s.clamp_min(0))[...,None]
    return F.linear(F.gelu(F.linear(a, P['core.age.0.weight'], P['core.age.0.bias'])), P['core.age.2.weight'], P['core.age.2.bias'])
def core_rest(P, hf, E, Mx, Mage, Mmask, agemode='kv', keep=False):
    x=hf+E; ag=agE(P, Mage); info={}
    for li,b in enumerate(['core.blocks.0.0','core.blocks.0.1']):
        n1=LNx(x,P,b+'.norm1'); x=x+mh(P,b+'.attn',n1,n1)
        n2=LNx(x,P,b+'.norm2')
        if agemode=='kv': x=x+mh(P,b+'.cross',n2,Mx+ag,Mmask)
        else: x=x+mh(P,b+'.cross',n2,Mx,Mmask,kadd=ag)
        n3=LNx(x,P,b+'.norm3')
        if keep: info[f'x{li}']=x; info[f'n3_{li}']=n3
        x=x+ffP(P,b+'.ffn',n3)
    sp=F.softplus(P['core.delta_in']); abar=torch.exp(-sp*torch.exp(P['core.a_in']))
    hn=abar*hf+sp*F.linear(E,P['core.b_in.weight'])+(x-(hf+E))
    c=F.linear(hn,P['core.c_out.weight']); n1=LNx(c,P,'core.coda.0.norm1'); c=c+mh(P,'core.coda.0.attn',n1,n1)
    y=c+ffP(P,'core.coda.0.ffn',LNx(c,P,'core.coda.0.norm3'))
    out=LNx(y.mean(1),P,'core.out_norm')
    return hn,y,out,info
def mem_for(ds, rows_by_stay, e_all):
    tsec=torch.as_tensor(np.asarray(ds['tsec']),dtype=torch.float32); N=len(tsec); L=max(len(r) for r in rows_by_stay)
    Mx=torch.zeros(N,L,192); Ma=torch.zeros(N,L); Mm=torch.zeros(N,L,dtype=torch.bool)
    for rr in rows_by_stay:
        rr=list(rr)
        for i,k in enumerate(rr):
            idx=rr[max(0,i-127):i+1]; n=len(idx)
            Mx[k,:n]=e_all[idx]; Ma[k,:n]=tsec[k]-tsec[idx]; Mm[k,:n]=True
    return Mx,Ma,Mm
'''
open('/work/lib2.py','a').write(code3); exec(code3)
Mx,Ma,Mm=mem_for(ds, ds['srows'], ds['e'])
for mode in ['kv','k']:
    hn,y,out,_=core_rest(PO, st['hf'], st['E'], Mx, Ma, Mm, agemode=mode)
    print(mode, 'hn err %.3g'%float((hn-st['hn']).abs().max()), 'y err %.3g'%float((y-r[0]).abs().max()), 'out err %.3g'%float((out-r[1]).abs().max()))
```