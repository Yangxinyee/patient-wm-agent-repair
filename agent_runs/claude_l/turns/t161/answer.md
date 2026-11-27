The core tensors match the structure in my `core_step` code: prelude and coda layers with attn, ffn, norm1 and norm3, and two block layers that add cross-attention with norm2. What is still unknown: the FFN/age activation (gelu or silu), whether the memory M includes the current bundle's e, the age units (seconds or hours), and whether the age encoding is added to keys only or to the memory content. I'll grid over these on two stays and compare predicted h and out with the recorded states. An exact match (error ~1e-5) gives a full replica. If no combination matches, the error pattern points to the remaining component, for example the number of heads.

```python
def run_stay_steps(s, opt, nmax=60):
    i0=row_start[s]; B_=BUN[s]; Ms=[]; ts=[]; eh=[]; eo=[]
    for k in range(len(B_)):
        e,u = enc_eu2(B_[k][1])
        if k>0:
            prev=torch.tensor(Xs[i0+k-1],dtype=T); out_p=prev[:192]; h_p=prev[192:960].reshape(4,192)
            dt=(B_[k][0]-B_[k-1][0])*3600.0
            if opt['m_incl']: Mt=torch.stack(Ms+[e]); ages=torch.tensor([(B_[k][0]-tj)*3600.0 for tj in ts+[B_[k][0]]],dtype=T)
            else: Mt=torch.stack(Ms) if Ms else None; ages=torch.tensor([(B_[k][0]-tj)*3600.0 for tj in ts],dtype=T) if Ms else None
            o2,y2,z2,_=core_step(W0,out_p,h_p,dt,e,u,Mt=Mt,ages=ages,opt=dict(age_unit=opt['au'],age_mode=opt['am']))
            eh.append(float((y2.reshape(-1)-torch.tensor(Xs[i0+k,192:960],dtype=T)).abs().max()))
            eo.append(float((o2-torch.tensor(Xs[i0+k,:192],dtype=T)).abs().max()))
        Ms.append(e); ts.append(B_[k][0])
        if k>=nmax: break
    return np.array(eh), np.array(eo)
res_opts=[]
for act_ in ['gelu','silu']:
    ACTMAP['core.']=act_
    for m_incl in [False, True]:
        for au in ['s','h']:
            for am in ['key','val']:
                op=dict(m_incl=m_incl,au=au,am=am)
                eh,eo = run_stay_steps(ks[0], op, nmax=30)
                res_opts.append((act_,m_incl,au,am,np.median(eh),np.median(eo)))
for r_ in sorted(res_opts,key=lambda r:r[4])[:8]: print(r_[:4], f"h med {r_[4]:.2e} out med {r_[5]:.2e}")
```