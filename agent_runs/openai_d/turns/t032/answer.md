The search reduced reconstruction RMSE from about 2.67 to 1.69, but that alone does not verify it. I’ll test alternative prior contexts and average many latent draws to estimate the remaining sampling error. I’ll also locate the treatment-history tables needed for a dose-specific residual correction.

```python
rh_prior_trials=[]
rh_vg=rc_mlp(rh_times["seconds_div10"],"gap","gelu")
rh_baseview=rh_bases["flow"]+rh_vg
for pb,pt,pa,mu_layout,reg in itertools.product(
        rh_bases,rh_times,("silu","gelu"),("split","interleaved"),
        ((6,3,"forecast_hours"),(2,3,"forecast_seconds"))):
    pp=rc_mlp(rh_bases[pb]+rc_mlp(rh_times[pt],"gap","gelu"),"z_prior",pa)
    mu=pp[:,:32] if mu_layout=="split" else pp[:,::2]
    q=rc_norm(rh_baseview+rc_lin(mu,"z_head")+W0["type_emb.weight"][1],"query_norm")
    nf=rh_register(*reg)
    score,rmse,_,_,_=rh_eval(q,nf,"gelu")
    rh_prior_trials.append((score,rmse,(pb,pt,pa,mu_layout,reg),pp,q,nf))
rh_prior_trials.sort(key=lambda x:x[0])
print("Best prior conventions:")
for rr in rh_prior_trials[:6]:
    print(rr[:3])
rh_mc_results=[]
rh_rng=torch.Generator().manual_seed(48201)
rh_eps=torch.randn((128,1000,32),generator=rh_rng)
for rr in rh_prior_trials[:6]:
    _,_,cc,pp,_,nf=rr
    mu=pp[:,:32] if cc[3]=="split" else pp[:,::2]
    ls=pp[:,32:] if cc[3]=="split" else pp[:,1::2]
    probs_batches=[]
    means_batches=[]
    for start in range(0,128,16):
        z=mu[None]+rh_eps[start:start+16]*torch.exp(ls.clamp(-8,3))[None]
        view=rh_baseview[None]+rc_lin(z,"z_head")
        qq=rc_norm(view+W0["type_emb.weight"][1],"query_norm")
        wc=qq+rh_fixed[None]+rc_lin(nf,"last_numeric")[None]
        hid=F.gelu(rc_lin(wc,"attr_heads.value_bin.0"))
        ppred=torch.softmax(rc_lin(hid,"attr_heads.value_bin.2")+rh_trans[None],-1)
        probs_batches.append(ppred)
        means_batches.append((ppred*rh_centres[None]).sum(-1))
    allp=torch.cat(probs_batches)
    allm=torch.cat(means_batches)
    mm=allm.mean(0)
    noise_var=allm.var(0,unbiased=True)/4
    mse=((mm-rh_tool)**2).mean().item()
    expected_noise=noise_var.mean().item()
    nrm_mse=(((mm-rh_tool)/rh_iqr)**2).mean().item()
    nrm_noise=(noise_var/rh_iqr**2).mean().item()
    result={"config":cc,"mean_rmse":np.sqrt(mse),"expected_four_draw_rmse":np.sqrt(expected_noise),
            "normalized_rmse":np.sqrt(nrm_mse),"normalized_noise":np.sqrt(nrm_noise),
            "excess_normalized_mse":nrm_mse-nrm_noise,
            "mean_log_sd":ls.mean().item()}
    rh_mc_results.append(result)
    print(result)
rh_chosen=min(rh_mc_results,key=lambda x:x["excess_normalized_mse"])
print("\n--- AVAILABLE PANELS ---")
for k,v in list(globals().items()):
    if isinstance(v,pd.DataFrame) and len(v)>10000:
        print(k,v.shape,list(v.columns)[:20])
print("\n--- DOSE RESIDUAL CACHE ---")
for drug,cc in dml_cache.items():
    print(drug,{k:(getattr(v,"shape",None) if hasattr(v,"shape") else type(v).__name__)
                for k,v in cc.items()})
print("\n--- VALUE CENTRE FORMAT ---")
print(str(VOCAB["value_bin_centres"])[:500])
with open("/work/notes.md","a") as f:
    f.write("\n## Run 26: latent integration check\n"+json.dumps(rh_mc_results,default=str)+"\n"
            "Original weights remain unchanged. Next: use dose innovations and paired factual residuals "
            "to construct a small statistical head correction, rather than amplify the existing dose direction.\n")
```