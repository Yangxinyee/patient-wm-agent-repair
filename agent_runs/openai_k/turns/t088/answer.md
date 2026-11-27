The bridge failed: useful intervention slopes required very large factual errors and unstable coefficients. The tiny carrier is not identifiable against normal state variation. I’ll test stronger carriers while monitoring factual NLL. Better held-out decoding within the fit budget would support amplification; otherwise this route should be rejected.

```python
t82 = time.time()
carrier_results82 = {}
diagnostics82 = []
saved_weights82 = {k: W0[k] for k in post80_weights}
saved_mlp82 = mlp0

def bridge_frontier82(pf, cfparts):
    dp = np.concatenate([v["P"]-pf[lookup80[v["indices"]]] for v in cfparts])
    mu = pf[train81,:-1].mean(axis=0)
    sd = np.maximum(pf[train81,:-1].std(axis=0),1e-5)
    xf = np.c_[(pf[:,:-1]-mu)/sd,np.ones(len(pf))]
    xd = np.c_[dp[:,:-1]/sd,np.zeros(len(dp))]
    af = xf[train81].T@xf[train81]/train81.sum()
    ad = xd[cftrain81].T@xd[cftrain81]/cftrain81.sum()
    cy = xd[cftrain81].T@Y81[cftrain81]/cftrain81.sum()
    result = {}
    for preservation in (0.01,0.1,1.0):
        bs = np.linalg.solve(ad+preservation*af+1e-5*reg81,cy)
        b = np.empty_like(bs)
        b[:-1]=bs[:-1]/sd[:,None]
        b[-1]=bs[-1]-mu@b[:-1]
        fp=pf[~train81]@b
        cp=dp[~cftrain81]@b
        yt=Y81[~cftrain81]
        slopes=[]
        for d in (0,1):
            sel=drug81[~cftrain81]==d
            slopes.append(float(np.sum(cp[sel,d]*yt[sel,d])/np.sum(yt[sel,d]**2)))
        result[preservation]=(b,dict(
            factual_RMS=np.sqrt(np.mean(fp**2,axis=0)).round(4).tolist(),
            intervention_slopes=np.round(slopes,4).tolist()))
    return result

try:
    mlp0=traced_mlp74
    for strength in (10.,100.,1000.):
        weights={k:torch.as_tensor(v,dtype=torch.float32).clone()
                 for k,v in post66_base_weights.items()}
        weights["z_post.2.weight"][:32] += torch.tensor(
            strength*arr71(carrier74)@coef80[:-1].T,dtype=torch.float32)
        weights["z_post.2.bias"][:32] += torch.tensor(
            strength*arr71(carrier74)@coef80[-1],dtype=torch.float32)
        W0.update(weights)
        pf,mf,prf=dataset80(idx80)
        pcparts=[]
        for old in CF81:
            pc,mc,prc=dataset80(old["indices"],u=old["u"])
            pcparts.append(dict(indices=old["indices"],P=pc,M=mc,probs=prc,
                                drug=old["drug"],factor=old["factor"]))
        front=bridge_frontier82(pf,pcparts)
        tp=np.take_along_axis(prf[rows80],bins80[...,None],axis=-1)[...,0]
        dnll=float(np.mean(np.log(np.maximum(truth0_80[mask80],1e-12))-
                           np.log(np.maximum(tp[mask80],1e-12))))
        diag=dict(strength=strength,dNLL=dnll,
                  mean_shift_RMS=np.sqrt(np.mean((mf-M0_80)**2,axis=0)).round(4).tolist(),
                  bridge={k:v[1] for k,v in front.items()})
        diagnostics82.append(diag)
        carrier_results82[strength]=dict(weights=weights,P=pf,M=mf,probs=prf,
                                        CF=pcparts,frontier=front)
        print(diag)
finally:
    W0.update(saved_weights82)
    mlp0=saved_mlp82

print("\nLATENT INPUT TENSORS")
for name in NAMES:
    if name.startswith(("z_in.","z_head.")):
        print(name,tuple(W0[name].shape),float(W0[name].norm()))
print("seconds:",round(time.time()-t82,1),"submitted edits:",edited())

with open("/work/notes.md","a") as f:
    f.write("\n## Run 82: stronger dose-carrier frontier\n"
            "- Small-carrier bridge failed held-out preservation/response trade-off; no candidate was submitted.\n"
            "- Tested 10x, 100x, and 1000x carrier strengths with local factual and held-out bridge diagnostics.\n"
            "- Results retained in carrier_results82/diagnostics82; scratch weights restored.\n"
            "- Forward passes: 4; Python runs: 82.\n")
```