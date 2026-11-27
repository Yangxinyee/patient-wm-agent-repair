The state request succeeded, giving us training-stay states without changing any weights. Next I’ll reconstruct the value head locally and check it against the GPU predictions. This lets us test closed-form head edits cheaply. Close agreement would validate the reconstruction; poor agreement would mean it is not yet safe to use for editing.

```python
import torch, torch.nn.functional as F
torch.set_num_threads(8)
torch.set_grad_enabled(False)
sTrain = np.load("/work/fwd_4/states_original.npz")
norm_meta = {(e["code"],e["attribute"]):e for e in VOCAB["numeric"]}
code_id = {e["name"]:e["id"] for e in VOCAB["codes"]}

print("=== CORRECTED DYNAMIC ALTERNATIVES ===")
dynamic_alternatives = {}
for label,drugletter,jj in [("G","I",0),("M","N",1)]:
    yr = np.array([0.] + [lp[(label,h)]["coef"][jj] for h in range(1,9)])
    ar = np.array([lp[(drugletter,h)]["coef"][jj] for h in range(8)])
    design2 = np.column_stack([yr[:-1],ar,np.r_[0.,ar[:-1]]])
    coeff2 = np.linalg.lstsq(design2,yr[1:],rcond=None)[0]
    kernel = np.empty(8)
    for h in range(8):
        kernel[h] = yr[h+1] - sum(kernel[h-l]*ar[l] for l in range(1,h+1))
    dynamic_alternatives[label] = {"arx":coeff2,"kernel":kernel}
    print(label,"ARX [phi, beta_now, beta_lag]",np.round(coeff2,4),
          "impulse kernel",np.round(kernel,3))

def lin0(v,prefix):
    ans = F.linear(v,W0[prefix+".weight"],W0.get(prefix+".bias"))
    return ans
def norm0(v,prefix):
    return F.layer_norm(v,(192,),W0[prefix+".weight"],W0[prefix+".bias"],1e-5)
def act0(v,kind):
    return {"gelu":F.gelu,"silu":F.silu,"relu":F.relu}[kind](v)
def mlp0(v,prefix,kind):
    return lin0(act0(lin0(v,prefix+".0"),kind),prefix+".2")

def reading_cache(records, states, factual=None):
    stids,sth = states["stay_id"],states["hour"]
    locations = {int(s):np.flatnonzero(stids==s) for s in np.unique(stids)}
    df = records[records.event.isin(["glucose_finger","glucose_lab","map_arterial","map_cuff"])
                 & records.stay_id.isin(locations)].copy()
    df = df.sort_values(["stay_id","hour","event"]).reset_index(drop=True)
    previous = np.full(len(df),-1,dtype=int)
    lastval = np.zeros(len(df)); lastage = np.zeros(len(df)); haslast = np.zeros(len(df))
    lastbin = np.zeros(len(df),dtype=int); targetbin=np.zeros(len(df),dtype=int)
    for sid0, ix in df.groupby("stay_id",sort=False).indices.items():
        loc=locations[int(sid0)]
        pos=np.searchsorted(sth[loc],df.loc[ix,"hour"].to_numpy(),side="left")-1
        previous[ix]=np.where(pos>=0,loc[np.maximum(pos,0)],-1)
        for event, subix in df.loc[ix].groupby("event",sort=False).groups.items():
            subix=np.array(list(subix))
            vals=df.loc[subix,"value"].to_numpy()
            hh=df.loc[subix,"hour"].to_numpy()
            meta=norm_meta[(event,"value")]
            targetbin[subix]=1+np.searchsorted(meta["fine_bin_edges"],vals,side="left")
            if len(subix)>1:
                ii=subix[1:]
                haslast[ii]=1
                lastval[ii]=np.clip((vals[:-1]-meta["median"])/meta["iqr"],-10,10)
                lastage[ii]=np.log1p((hh[1:]-hh[:-1])*3600)/10
                lastbin[ii]=1+np.searchsorted(meta["fine_bin_edges"],vals[:-1],side="left")
    good=previous>=0
    df=df.loc[good].reset_index(drop=True)
    previous=previous[good]
    arr=torch.from_numpy(states["x"][previous].astype(np.float32))
    dt=torch.tensor(((df.hour.to_numpy()-sth[previous])*3600).astype(np.float32))[:,None]
    h=arr[:,192:960].reshape(-1,4,192)
    hflow=h*torch.exp(-dt[:,:,None]*torch.exp(W0["core.flow.a"])[None,None,:])
    base=arr[:,:192]+norm0(hflow.mean(1),"core.out_norm")
    centers=np.stack([np.r_[0.,VOCAB["value_bin_centres"][e]] for e in df.event]).astype(np.float32)
    codes=torch.tensor([code_id[e] for e in df.event])
    result=dict(df=df,base=base,dt=dt,codes=codes,
                haslast=torch.tensor(haslast[good],dtype=torch.float32),
                lastval=torch.tensor(lastval[good],dtype=torch.float32),
                lastage=torch.tensor(lastage[good],dtype=torch.float32),
                lastbin=torch.tensor(lastbin[good],dtype=torch.long),
                targetbin=torch.tensor(targetbin[good],dtype=torch.long),
                centers=torch.tensor(centers),previous=previous)
    if factual is not None:
        merged=df[["stay_id","hour","event"]].merge(factual,on=["stay_id","hour","event"],how="left")
        result["gpu_mean"]=merged.predicted_mean.to_numpy()
        result["gpu_nll"]=merged.nll.to_numpy()
    return result

cacheA=reading_cache(tuning,sA,fA)
cacheTrain=reading_cache(train,sTrain)
def head0(cache,kind="gelu",gapmode="scaled",typeidx=1,flagcol=3,statusrow=0,subset=None):
    ii = slice(None) if subset is None else subset
    dt=cache["dt"][ii]
    gapinp=torch.log1p(dt)/(10 if gapmode=="scaled" else 1)
    ctx=cache["base"][ii]+mlp0(gapinp,"gap",kind)
    zm=mlp0(ctx,"z_prior",kind)[:,:32]
    view=ctx+lin0(zm,"z_head")
    q=norm0(view+W0["type_emb.weight"][typeidx],"query_norm")
    lastnums=torch.zeros((len(q),8))
    lastnums[:,0]=cache["haslast"][ii]
    lastnums[:,1]=cache["lastage"][ii]
    lastnums[:,2]=cache["lastval"][ii]
    if flagcol is not None:
        lastnums[:,flagcol]=cache["haslast"][ii]
    codes=cache["codes"][ii]; bins=cache["lastbin"][ii]
    wc=q+W0["embed.code.weight"][codes]+lin0(lastnums,"last_numeric")
    wc=wc+W0["embed.bins.value.weight"][bins]+W0["embed.bins.rate.weight"][0]
    if statusrow is not None:
        wc=wc+W0["last_status.weight"][statusrow]
    hidden=act0(lin0(wc,"attr_heads.value_bin.0"),kind)
    logits=lin0(hidden,"attr_heads.value_bin.2")+W0["embed_transition.value_bin.weight"][codes*17+bins]
    probs=logits.softmax(-1)
    means=(probs*cache["centers"][ii]).sum(-1)
    nll=-probs[torch.arange(len(probs)),cache["targetbin"][ii]].clamp_min(1e-12).log()
    return means.numpy(),nll.numpy(),hidden,wc,logits

subprobe=np.flatnonzero((cacheA["df"].hour.to_numpy()>=1)&(cacheA["haslast"].numpy()>0))[::4]
scale=np.array([norm_meta[(e,"value")]["iqr"] for e in cacheA["df"].event])[subprobe]
reconstruction_scores=[]
for kind in ["gelu","silu","relu"]:
    for gapmode in ["scaled","unscaled"]:
        for typeidx in [0,1,3]:
            for flagcol in [None,3,4]:
                for statusrow in [None,0,1]:
                    args=(kind,gapmode,typeidx,flagcol,statusrow)
                    mm,nn,*_=head0(cacheA,*args,subset=subprobe)
                    rmse=np.sqrt(np.mean(((mm-cacheA["gpu_mean"][subprobe])/scale)**2))
                    nerr=np.sqrt(np.mean((nn-cacheA["gpu_nll"][subprobe])**2))
                    reconstruction_scores.append((rmse,nerr,args))
reconstruction_scores.sort(key=lambda a:a[0])
print("\n=== BEST LOCAL HEAD RECONSTRUCTIONS ===")
for result in reconstruction_scores[:12]:
    print(result)
print("cache sizes:",len(cacheA["df"]),len(cacheTrain["df"]))
with open("/work/notes.md","a") as f:
    f.write("\n## Local head reconstruction (run 10, forward 4)\n"
            "- Obtained original states for 100 training stays (4,314 bundles).\n"
            "- Computing a no-gradient local value-head reconstruction and checking it against original GPU predictions.\n"
            "- Corrected local-projection dictionary access and calculated ARX/deconvolved alternatives.\n")
```
