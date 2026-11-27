The simple head reconstruction is not accurate enough yet: its mean errors are roughly 13% of a reading’s normalisation scale. That could reflect an incorrect register convention or the head’s sampled latent variable. I’ll separate those possibilities next. If latent sampling accounts for the remaining error, the reconstruction can support candidate edits; otherwise I’ll correct it first.

```python
print("=== LATENT AND TIME-FLOW SCALES ===")
print("flow decay / second:",np.quantile(torch.exp(W0["core.flow.a"]).numpy(),[0,.25,.5,.75,1]))
probe_ctx=cacheA["base"]+mlp0(torch.log1p(cacheA["dt"])/10,"gap","gelu")
probe_z=mlp0(probe_ctx,"z_prior","gelu")
print("prior mean quantiles",np.quantile(probe_z[:,:32].numpy(),[0,.1,.5,.9,1]))
print("prior log-sd quantiles",np.quantile(probe_z[:,32:].numpy(),[0,.1,.5,.9,1]))

def head_parts(cache,kind="gelu",flagcol=3,ageunit="seconds",subset=None):
    ii=slice(None) if subset is None else subset
    ctx=cache["base"][ii]+mlp0(torch.log1p(cache["dt"][ii])/10,"gap",kind)
    zp=mlp0(ctx,"z_prior",kind)
    lastnums=torch.zeros((len(ctx),8))
    lastnums[:,0]=cache["haslast"][ii]
    if ageunit=="seconds":
        lastnums[:,1]=cache["lastage"][ii]
    elif ageunit=="hours":
        lastnums[:,1]=torch.log1p(torch.expm1(cache["lastage"][ii]*10)/3600)/10
    elif ageunit=="minutes":
        lastnums[:,1]=torch.log1p(torch.expm1(cache["lastage"][ii]*10)/60)/10
    lastnums[:,2]=cache["lastval"][ii]
    if flagcol is not None:
        lastnums[:,flagcol]=cache["haslast"][ii]
    codes=cache["codes"][ii]; bins=cache["lastbin"][ii]
    register=W0["embed.code.weight"][codes]+lin0(lastnums,"last_numeric")
    register=register+W0["embed.bins.value.weight"][bins]+W0["embed.bins.rate.weight"][0]
    trans=W0["embed_transition.value_bin.weight"][codes*17+bins]
    return ctx,zp,register,trans

def head_mc(cache,kind="gelu",flagcol=3,ageunit="seconds",nsamples=32,subset=None,seed=73):
    ctx,zp,register,trans=head_parts(cache,kind,flagcol,ageunit,subset)
    gen=torch.Generator().manual_seed(seed)
    eps=torch.randn((nsamples,len(ctx),32),generator=gen)
    z=zp[None,:,:32]+eps*torch.exp(zp[None,:,32:].clamp(-6,2))
    view=ctx[None]+lin0(z,"z_head")
    q=norm0(view+W0["type_emb.weight"][1],"query_norm")
    wc=q+register[None]
    hidden=act0(lin0(wc,"attr_heads.value_bin.0"),kind)
    logits=lin0(hidden,"attr_heads.value_bin.2")+trans[None]
    probs=logits.softmax(-1)
    return probs,hidden,wc

register_scores=[]
for kind in ["gelu","silu"]:
    for flagcol in [None,3,4,5,6,7]:
        for ageunit in ["seconds","minutes","hours"]:
            pp,_,_=head_mc(cacheA,kind,flagcol,ageunit,nsamples=8,subset=subprobe)
            pbar=pp.mean(0)
            mm=(pbar*cacheA["centers"][subprobe]).sum(-1).numpy()
            nn=-pbar[torch.arange(len(subprobe)),cacheA["targetbin"][subprobe]].clamp_min(1e-12).log().numpy()
            err=np.sqrt(np.mean(((mm-cacheA["gpu_mean"][subprobe])/scale)**2))
            nerr=np.sqrt(np.mean((nn-cacheA["gpu_nll"][subprobe])**2))
            register_scores.append((err,nerr,(kind,flagcol,ageunit)))
register_scores.sort(key=lambda a:a[0])
print("\n=== REGISTER / MONTE CARLO CHECK ===")
for row in register_scores[:10]:
    print(row)

best_head_spec=register_scores[0][2]
pcheck,hcheck,wcheck=head_mc(cacheA,*best_head_spec,nsamples=64,subset=subprobe)
pbar=pcheck.mean(0)
meanbar=(pbar*cacheA["centers"][subprobe]).sum(-1).numpy()
groupmeans=(pcheck.reshape(16,4,len(subprobe),17).mean(1)*cacheA["centers"][subprobe]).sum(-1).numpy()
monte_err=np.sqrt(np.mean(((groupmeans-meanbar[None])/scale[None])**2))
observed_err=np.sqrt(np.mean(((meanbar-cacheA["gpu_mean"][subprobe])/scale)**2))
print("best spec:",best_head_spec)
print("GPU-vs-local mean error:",observed_err,"expected MC4 variation:",monte_err)
for event in ["glucose_finger","glucose_lab","map_arterial","map_cuff"]:
    vm=cacheA["df"].event.to_numpy()[subprobe]==event
    dif=meanbar[vm]-cacheA["gpu_mean"][subprobe][vm]
    print(event,"mean difference",round(float(dif.mean()),3),
          "RMSE",round(float(np.sqrt(np.mean(dif**2))),3),
          "MC4 SD",round(float(np.sqrt(np.mean((groupmeans[:,vm]-meanbar[None,vm])**2))),3))
with open("/work/notes.md","a") as f:
    f.write("\n## Head reconstruction validation (run 11)\n"
            "- Simple prior-mean reconstruction had nontrivial error; testing register conventions and latent integration.\n"
            "- ARX alternatives: glucose phi .7807, beta_now -1.8381, beta_lag -.7382; "
            "MAP phi .5116, beta_now 3.2133, beta_lag 1.3739. These remain conditional estimates.\n")
```
