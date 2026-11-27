The register really has eight inputs, and the best previous layouts still disagree with the tool beyond estimated sampling noise. I’ll systematically test the time, activation, and register conventions without editing weights. A reconstruction whose error is consistent with four-draw noise would make a head-level repair trustworthy.

```python
import itertools, time
import torch.nn.functional as F
torch.set_num_threads(8)
tstart=time.time()
rh_meta=head_meta.iloc[:1000].copy().reset_index(drop=True)
rh_x=torch.as_tensor(states0["x"][rh_meta.state_idx.to_numpy()],dtype=torch.float32)
rh_out=rh_x[:,:192]
rh_h=rh_x[:,192:960].reshape(-1,4,192)
rh_dt=torch.tensor(rh_meta.dt.to_numpy(),dtype=torch.float32)[:,None]
rh_flow=rh_h*torch.exp(-rh_dt[:,:,None]*torch.exp(W0["core.flow.a"])[None,None,:])
rh_bases={
    "flow":rh_out+rc_norm(rh_flow.mean(1),"core.out_norm"),
    "no_flow":rh_out+rc_norm(rh_h.mean(1),"core.out_norm"),
    "out_only":rh_out
}
rh_times={
    "seconds_div10":torch.log1p(rh_dt)/10,
    "seconds":torch.log1p(rh_dt),
    "hours_div10":torch.log1p(rh_dt/3600)/10,
    "hours":torch.log1p(rh_dt/3600)
}
rh_codes=torch.tensor([code_ids[x] for x in rh_meta.event],dtype=torch.long)
rh_norm={ (x["code"],x["attribute"]):x for x in VOCAB["numeric"] }
rh_lb=[]
for row in rh_meta.itertuples():
    n=rh_norm[(row.event,"value")]
    raw=row.last_z*n["iqr"]+n["median"]
    rh_lb.append(1+np.searchsorted(n["fine_bin_edges"],raw,side="left") if row.has_last else 0)
rh_lb=torch.tensor(rh_lb,dtype=torch.long)
rh_centres=centres.clone()
assert tuple(rh_centres.shape)==(1000,17)
rh_fixed=(W0["embed.code.weight"][rh_codes]+W0["embed.bins.value.weight"][rh_lb]
          +W0["embed.bins.rate.weight"][0]+W0["last_status.weight"][0])
rh_trans=W0["embed_transition.value_bin.weight"][rh_codes*17+rh_lb]
rh_tool=torch.tensor(rh_meta.tool_mean.to_numpy(),dtype=torch.float32)
rh_iqr=torch.tensor([rh_norm[(x,"value")]["iqr"] for x in rh_meta.event],dtype=torch.float32)
rh_has=torch.tensor(rh_meta.has_last.to_numpy(),dtype=torch.float32)
rh_last=torch.tensor(rh_meta.last_z.to_numpy(),dtype=torch.float32)
def rh_register(zpos,flagpos,age_kind):
    nf=torch.zeros((len(rh_meta),8))
    nf[:,0]=rh_has
    if age_kind!="zero":
        age=torch.tensor(rh_meta.age.to_numpy(),dtype=torch.float32)
        if age_kind.startswith("state"):
            age=(age-rh_dt[:,0]).clamp_min(0)
        if age_kind.endswith("hours"):
            age=age/3600
        nf[:,1]=rh_has*torch.log1p(age)/10
    nf[:,zpos]=rh_last
    if flagpos is not None:
        nf[:,flagpos]=rh_has
    return nf
rh_regs=[rh_register(2,3,"forecast_seconds"),rh_register(6,3,"zero")]
def rh_eval(q,nf,head_act):
    wc=q+rh_fixed+rc_lin(nf,"last_numeric")
    hidden=rc_act(rc_lin(wc,"attr_heads.value_bin.0"),head_act)
    probs=torch.softmax(rc_lin(hidden,"attr_heads.value_bin.2")+rh_trans,dim=-1)
    means=(probs*rh_centres).sum(-1)
    err=(means-rh_tool)/rh_iqr
    return float(torch.sqrt((err*err).mean())),float(torch.sqrt(((means-rh_tool)**2).mean())),hidden,probs,means
rh_contexts=[]
for base_key,view_t,prior_t,gap_act,prior_act in itertools.product(
        rh_bases,rh_times,rh_times,("silu","gelu"),("silu","gelu")):
    base=rh_bases[base_key]
    vg=rc_mlp(rh_times[view_t],"gap",gap_act)
    pg=rc_mlp(rh_times[prior_t],"gap",gap_act)
    zp=rc_mlp(base+pg,"z_prior",prior_act)
    view=base+vg+rc_lin(zp[:,:32],"z_head")
    q=rc_norm(view+W0["type_emb.weight"][1],"query_norm")
    scores=[rh_eval(q,nf,"silu")[0] for nf in rh_regs]
    rh_contexts.append((min(scores),(base_key,view_t,prior_t,gap_act,prior_act),q,zp,base+vg))
rh_contexts.sort(key=lambda x:x[0])
print("Best contexts with two initial registers:")
for score,cfg,_,_,_ in rh_contexts[:6]:
    print(round(score,6),cfg)
rh_candidates=[]
for _,cfg,q,zp,baseview in rh_contexts[:4]:
    for zpos,flagpos,age_kind,head_act in itertools.product(
            range(2,8),[None,2,3,4,5,6,7],
            ("zero","forecast_hours","forecast_seconds","state_hours","state_seconds"),
            ("silu","gelu")):
        if flagpos==zpos:
            continue
        nf=rh_register(zpos,flagpos,age_kind)
        score,rmse,hidden,probs,means=rh_eval(q,nf,head_act)
        rh_candidates.append((score,rmse,cfg,(zpos,flagpos,age_kind,head_act)))
rh_candidates.sort(key=lambda x:x[0])
print("\nBest complete configurations:")
for x in rh_candidates[:12]:
    print(x)
rh_best=rh_candidates[0]
_,_,rh_cfg,rh_rcfg=rh_best
rh_context=next(x for x in rh_contexts if x[1]==rh_cfg)
rh_nf=rh_register(*rh_rcfg[:3])
_,_,rh_hidden,rh_probs,rh_means=rh_eval(rh_context[2],rh_nf,rh_rcfg[3])
rh_diag=rh_meta.assign(err=(rh_means-rh_tool).numpy())
print("\nError by event:")
print(rh_diag.groupby("event").err.agg(["count","mean",lambda x:np.sqrt(np.mean(x*x))]))
print("Elapsed seconds:",round(time.time()-tstart,2))
with open("/work/notes.md","a") as f:
    f.write("\n## Run 25: systematic local-head check\n"
            f"Best deterministic reconstruction: {rh_best}. No weight edits. "
            "Must still compare to latent sampling noise before trusting a head edit.\n")
```