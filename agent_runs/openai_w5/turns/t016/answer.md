Latent sampling explains only a small part of the reconstruction error. The remaining mismatch appears to involve the eight-column last-reading register, whose full ordering is not specified in the architecture description. I’ll test plausible orderings using the original predictions, without changing weights. A sharp reduction in error would identify the missing convention.

```python
import itertools
layout_probe=subprobe[::2]
layout_scale=torch.tensor(np.array([norm_meta[(e,"value")]["iqr"]
                                  for e in cacheA["df"].event])[layout_probe],dtype=torch.float32)
layout_gpu=torch.tensor(cacheA["gpu_mean"][layout_probe],dtype=torch.float32)
layouts=[]
for flags in itertools.combinations(range(8),2):
    remaining=[c for c in range(8) if c not in flags]
    for vcol in remaining:
        for agecol in remaining:
            if vcol!=agecol:
                layouts.append((vcol,agecol,flags))

layout_scores=[]
with threadpool_limits(limits=8):
    for kind in ["gelu","silu","relu"]:
        ctx=cacheA["base"][layout_probe]+mlp0(torch.log1p(cacheA["dt"][layout_probe])/10,"gap",kind)
        zp=mlp0(ctx,"z_prior",kind)
        q=norm0(ctx+lin0(zp[:,:32],"z_head")+W0["type_emb.weight"][1],"query_norm")
        codes=cacheA["codes"][layout_probe]; bins=cacheA["lastbin"][layout_probe]
        basewc=q+W0["embed.code.weight"][codes]+W0["last_numeric.bias"]
        basewc=basewc+W0["embed.bins.value.weight"][bins]+W0["embed.bins.rate.weight"][0]
        trans=W0["embed_transition.value_bin.weight"][codes*17+bins]
        for ageunit in ["seconds","minutes","hours"]:
            age=cacheA["lastage"][layout_probe]
            if ageunit!="seconds":
                age=torch.log1p(torch.expm1(age*10)/(60 if ageunit=="minutes" else 3600))/10
            for start0 in range(0,len(layouts),32):
                batch=layouts[start0:start0+32]
                vc=torch.stack([W0["last_numeric.weight"][:,a] for a,b,c in batch])
                ac=torch.stack([W0["last_numeric.weight"][:,b] for a,b,c in batch])
                fc=torch.stack([W0["last_numeric.weight"][:,list(c)].sum(-1) for a,b,c in batch])
                wc=basewc[None]+cacheA["lastval"][layout_probe][None,:,None]*vc[:,None]
                wc=wc+age[None,:,None]*ac[:,None]+cacheA["haslast"][layout_probe][None,:,None]*fc[:,None]
                hidden=act0(lin0(wc,"attr_heads.value_bin.0"),kind)
                probs=(lin0(hidden,"attr_heads.value_bin.2")+trans[None]).softmax(-1)
                means=(probs*cacheA["centers"][layout_probe][None]).sum(-1)
                errs=(((means-layout_gpu[None])/layout_scale[None])**2).mean(-1).sqrt().numpy()
                for err,layout in zip(errs,batch):
                    layout_scores.append((float(err),kind,ageunit,layout))
layout_scores.sort()
print("=== BEST REGISTER LAYOUTS ===")
for row in layout_scores[:15]:
    print(row)

def head_mc_layout(cache,kind,ageunit,layout,nsamples=16,subset=None,seed=73):
    ii=slice(None) if subset is None else subset
    ctx=cache["base"][ii]+mlp0(torch.log1p(cache["dt"][ii])/10,"gap",kind)
    zp=mlp0(ctx,"z_prior",kind)
    gen=torch.Generator().manual_seed(seed)
    eps=torch.randn((nsamples,len(ctx),32),generator=gen)
    z=zp[None,:,:32]+eps*torch.exp(zp[None,:,32:].clamp(-6,2))
    q=norm0(ctx[None]+lin0(z,"z_head")+W0["type_emb.weight"][1],"query_norm")
    vcol,agecol,flagcols=layout
    nums=torch.zeros((len(ctx),8))
    nums[:,vcol]=cache["lastval"][ii]
    age=cache["lastage"][ii]
    if ageunit!="seconds":
        age=torch.log1p(torch.expm1(age*10)/(60 if ageunit=="minutes" else 3600))/10
    nums[:,agecol]=age
    for c in flagcols:
        nums[:,c]=cache["haslast"][ii]
    codes=cache["codes"][ii]; bins=cache["lastbin"][ii]
    reg=W0["embed.code.weight"][codes]+lin0(nums,"last_numeric")
    reg=reg+W0["embed.bins.value.weight"][bins]+W0["embed.bins.rate.weight"][0]
    wc=q+reg[None]
    hidden=act0(lin0(wc,"attr_heads.value_bin.0"),kind)
    logits=lin0(hidden,"attr_heads.value_bin.2")+W0["embed_transition.value_bin.weight"][codes*17+bins][None]
    return logits.softmax(-1),hidden,wc,logits

best_layout_spec=layout_scores[0][1:]
pp,hh,ww,ll=head_mc_layout(cacheA,*best_layout_spec,nsamples=32,subset=subprobe)
pbar=pp.mean(0)
mm=(pbar*cacheA["centers"][subprobe]).sum(-1).numpy()
nn=-pbar[torch.arange(len(subprobe)),cacheA["targetbin"][subprobe]].clamp_min(1e-12).log().numpy()
print("\nbest layout spec:",best_layout_spec)
print("MC32 mean error",np.sqrt(np.mean(((mm-cacheA["gpu_mean"][subprobe])/scale)**2)),
      "NLL error",np.sqrt(np.mean((nn-cacheA["gpu_nll"][subprobe])**2)))
for event in ["glucose_finger","glucose_lab","map_arterial","map_cuff"]:
    vm=cacheA["df"].event.to_numpy()[subprobe]==event
    print(event,"mean bias",round(float((mm-cacheA["gpu_mean"][subprobe])[vm].mean()),3),
          "RMSE",round(float(np.sqrt(np.mean((mm-cacheA["gpu_mean"][subprobe])[vm]**2))),3))
```
