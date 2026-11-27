Raising κ to 4 changed almost nothing in the rollout. The replica's readouts are almost identical to the original model's tool readouts (row correlations 0.82–0.92):

| Edit | κ = 4 replica | Original (tool) |
|---|---|---|
| Insulin ×2 | −3.14 | −3.25 |
| Insulin ×½ | +1.84 | +1.77 |
| NE ×2 | +0.62 | +0.62 |
| NE ×½ | −0.88 | −0.82 |

Teacher forcing did shift predictions (finger sd 0.6), so the injection is active for ordinary small residuals. Under the edits the residual is large. With κ = 4 the z token moves by about 4 × 1.7 × 9.8 ≈ 65, more than three times |u − u_empty|.

My hypothesis is that after the prelude's LayerNorm such a large displacement changes the z token's attention key. The z token then stops being attended (or the effect saturates), so the large edit residuals have no effect.

Next I'll test this directly in the replica. At insulin bundles from 10 tuning stays I'll set the z token to an exact copy of u plus a × (insulin dose tangent), with a from −40 to +40. I'll record:
- the change in the next-hour finger-glucose prediction;
- the attention weight the slots give the z token.

**Confirmed** if the response is roughly linear for small a, then flattens or reverses as the attention weight collapses. **Refuted** if it stays linear and large, in which case the problem is in the residual computation instead.

```python
def probe_stay(W, B_, avals, cdir, drug=8, code=6, maxb=12):
    times=[b[0] for b in B_]
    out=torch.tensor(X0row[:192],dtype=T)[None]; h=torch.tensor(X0row[192:960],dtype=T).reshape(1,4,192)
    ea,ua=split_fix(B_[0][1]); Ms=[set_enc(W,ea,'encode')]; ts=[times[0]]; reg={}
    for a in ea:
        if a['code'] in MEASC and a['kind']==0: reg[a['code']]=(times[0],a['num'][0],a['vb'])
    A=torch.tensor(avals,dtype=T)[:,None]*torch.tensor(cdir,dtype=T)[None]; nb=len(avals); res=[]
    for k in range(1,len(B_)):
        t=times[k]; dt=(t-times[k-1])*3600.0; atoms=B_[k][1]; ea,ua=split_fix(atoms)
        for a in ea:
            if a['code'] in MEASC and a['kind']==0: reg[a['code']]=(t,a['num'][0],a['vb'])
        e=set_enc(W,ea,'encode'); u=set_enc(W,ua,'encode_action') if ua else u_empty
        Ms.append(e); ts.append(t); Mt=torch.stack(Ms)[None]; ages=torch.tensor([(t-tj)*3600.0 for tj in ts],dtype=T)
        has=any(a['code']==drug for a in ua) if ua else False
        if has and k+1<len(B_) and len(res)<maxb and code in reg:
            zf=lambda ctx,e_,u_,info: u_+A
            o2,h2,_=bstep_z(W,out.repeat(nb,1),h.repeat(nb,1,1),dt,e[None].repeat(nb,1),u[None].repeat(nb,1),Mt,ages,zfn=zf)
            dtn=(times[k+1]-t)*3600.0; views=bview(W,o2,h2,dtn,zs=torch.zeros(nb,1,32,dtype=T))
            tl,zl,vl=reg[code]; age=(times[k+1]-tl)*3600.0
            p=bvalue_probs(W,views,code,torch.ones(nb,dtype=T),torch.full((nb,),age,dtype=T),torch.full((nb,),zl,dtype=T),torch.full((nb,),vl,dtype=torch.long))
            mu=(p[:,1:]*torch.tensor(CENd[code],dtype=T)).sum(1).numpy(); res.append(mu-mu[list(avals).index(0.0)])
        out,h,_=bstep_z(W,out,h,dt,e[None],u[None],Mt,ages,zfn=lambda ctx,e_,u_,info: u_)
    return res
av=[-40.,-20.,-10.,-5.,0.,5.,10.,20.,40.]
Wd=build_W({8:0.0,9:0.0})
allr=[]
for s in keys[:10]: allr+=probe_stay(Wd,BUN[s],av,P[8]['tan'])
allr=np.array(allr); print('n',len(allr)); print('a     ',av); print('mean dG',np.round(allr.mean(0),3)); print('sd   dG',np.round(allr.std(0),3))
```