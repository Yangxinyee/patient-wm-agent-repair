My replica rollout reproduces the tool's dose_edit readout for the original model on all 100 stays, within sampling noise:

| Edit | Replica | Tool | Row correlation |
|---|---|---|---|
| insulin x2 | −3.46 | −3.25 | 0.93 |
| insulin x1/2 | +1.60 | +1.77 | 0.91 |
| NE x2 | +0.58 | +0.62 | 0.85 |
| NE x1/2 | −0.75 | −0.82 | 0.95 |

So I can now score candidate edits offline and save forward passes for confirmation. I'll record this. Next I'll measure how the original model's response to the dose splits between its two pathways. One is the z posterior, which sees both ctx and u, so it can carry a dose surprise. The other is the u token in the prelude, which carries the raw dose. I'll use teacher forcing: add +1 U/h to one insulin record and track the predicted glucose_finger over the next 12 hours, with z frozen or u frozen in turn. If most of the effect goes through z, amplifying a dose-residual direction in z is the natural edit. If it goes through the u token, the edit has to build the residual inside z_post itself.

```python
with open('/work/notes.md','a') as f:
    f.write("""
- ROLLOUT REPLICA (run 169-170; /work/wm2.py + /work/wm2_rollout.py, rollout(W,B_,conds,npaths,seed)): values drawn by inverse-CDF
  from value head, value = bin centre; reproduces fwd_1 tool readout: ins x2 -3.46 (tool -3.25, row corr .93), x1/2 1.60 (1.77),
  NE x2 .58 (.62), NE x1/2 -.75 (-.82). ~0.37 s/stay for 5 conds x 20 paths. => can evaluate edits OFFLINE.
""")
def tf_run(W, B_, mod=None):
    """teacher-forced run, B=1. mod: dict k-> dict(u=..., z_fix=..., zdelta=...). returns per-bundle states list and pred means for readings"""
    times=[b[0] for b in B_]; Ms=[]; ts=[]; out=None; h=None; preds={}
    for k in range(len(B_)):
        ea,ua=split_fix(B_[k][1]); e=set_enc(W,ea,'encode') if ea else set_enc(W,[],'encode'); u=set_enc(W,ua,'encode_action') if ua else u_empty
        if k==0:
            out=torch.tensor(X0row[:192],dtype=T)[None]; h=torch.tensor(X0row[192:960],dtype=T).reshape(1,4,192)
        else:
            dt=(times[k]-times[k-1])*3600.0
            # predictions for readings in bundle k
            views=bview(W,out,h,dt,zs=torch.zeros(1,1,32,dtype=T))
            for a in ea:
                if a['code'] in MEASC and a['kind']==0:
                    rf=reg_feats2(B_,k,a['code'],'cur')
                    p=bvalue_probs(W,views,a['code'],torch.tensor([rf['has']],dtype=T),torch.tensor([rf['age']],dtype=T),torch.tensor([rf['z']],dtype=T),torch.tensor([rf['vb']]))
                    preds[(round(times[k],3),a['code'])]=float((p[0,1:]*torch.tensor(CENd[a['code']],dtype=T)).sum())
            Ms_=torch.stack(Ms+[e])[None]; ages=torch.tensor([(times[k]-tj)*3600.0 for tj in ts+[times[k]]],dtype=T)
            m=(mod or {}).get(k,{})
            out,h,_=bstep_mod(W,out,h,dt,e[None],u[None],Ms_,ages,m)
        Ms.append(e); ts.append(times[k])
    return preds
def bstep_mod(W,out,h,dt,e,u,Mt,ages,m):
    hf=h*torch.exp(-dt*torch.exp(W['core.flow.a']))
    g=bmlp(W,torch.tensor([[math.log(dt)/10]],dtype=T),'gap')
    ctx=out+bln(W,hf.mean(1),'core.out_norm')+g
    uz=m.get('u_z',u); utok=m.get('u_tok',u)
    z=bmlp(W,torch.cat([ctx,e,uz],-1),'z_post')[:,:32]
    if 'zdelta' in m: z=z+m['zdelta']
    tok=torch.cat([hf+W['core.slots'],e[:,None],utok[:,None],blin(W,z,'z_in')[:,None]],1)
    tok=bself(W,tok,'core.prelude.0'); E=bln(W,tok[:,:4],'core.prelude_norm')
    ae=bmlp(W,torch.log1p(ages)[:,None],'core.age'); Mk=Mt+ae[None]; x=hf+E
    for L in ['core.blocks.0.0','core.blocks.0.1']:
        a_=bln(W,x,L+'.norm1'); x=x+bmha(W,a_,a_,L+'.attn')
        x=x+bmha(W,bln(W,x,L+'.norm2'),Mk,L+'.cross'); x=x+bmlp(W,bln(W,x,L+'.norm3'),L+'.ffn')
    sp=F.softplus(W['core.delta_in']); abar=torch.exp(-sp*torch.exp(W['core.a_in']))
    h2=abar*hf+sp*(E@W['core.b_in.weight'].T)+(x-(hf+E))
    y=bself(W,h2@W['core.c_out.weight'].T,'core.coda.0')
    return bln(W,y.mean(1),'core.out_norm'), y, z
# pathway test on several stays: pick insulin bundles at hour ~8-10, +1 U/h
def ins_bundles(B_): return [k for k,(t,al) in enumerate(B_) if any(a['code']==8 and a['kind']==0 for a in al)]
resp={'both':[], 'z_only':[], 'u_only':[]}
for s in ks[:25]:
    B_=BUN[s]; ib=[k for k in ins_bundles(B_) if 6<=B_[k][0]<=10]
    if not ib: continue
    k=ib[0]; base=tf_run(W0,B_)
    ua=[a for a in B_[k][1] if a['act'] and a['kind']==0]
    def u_with(f_add):
        ua2=[]
        for a in ua:
            if a['code']==8:
                nm='insulin'; d0=a['num'][4]*NUMV[(nm,'rate')]['iqr']+NUMV[(nm,'rate')]['median']; f=(d0+f_add)/d0; ua2.append(edit_atom(a,f))
            else: ua2.append(a)
        return set_enc(W0,[fix_atom2(a) for a in ua2],'encode_action')[None]
    u_new=u_with(1.0); u_old=set_enc(W0,[fix_atom2(a) for a in ua],'encode_action')[None]
    for nm_,m in [('both',dict(u_z=u_new,u_tok=u_new)),('z_only',dict(u_z=u_new,u_tok=u_old)),('u_only',dict(u_z=u_old,u_tok=u_new))]:
        p2=tf_run(W0,B_,{k:m}); t0_=B_[k][0]
        r_=[p2.get((round(t0_-0.0833+L,3),6),np.nan)-base.get((round(t0_-0.0833+L,3),6),np.nan) for L in range(1,13)]
        resp[nm_].append(r_)
for nm_,v in resp.items(): print(nm_, len(v), np.round(np.nanmean(np.array(v),0),3))
```