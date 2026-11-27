The input-level fit nearly doubled insulin sensitivity at negligible estimated factual cost, but did little for norepinephrine. I’ll check whether latent sampling makes that correction noisy and whether two head units can be repurposed safely. Low sampled loss and small compression error would justify a GPU trial.

```python
print("=== LATENT HEAD TENSORS ===")
for r125_name in NAMES:
    if r125_name.startswith(("core.z_prior", "core.z_head", "query_norm", "type_emb")):
        print(r125_name, SHAPES[r125_name])
print("Reconstruction constants:",
      [repr(v)[:90] for v in reconstruct_reading_head.__code__.co_consts
       if isinstance(v,(str,int,float,tuple))])

print("\n=== INSULIN-ONLY CORRECTION, BEFORE COMPRESSION ===")
r125_coef = r124_models[(0.01,1e-7)]["insulin"]
r125_tau = (r124_F@r125_coef)*r124_family
for r125_gain in [1.,1.5,2.,3.]:
    r125_Pn = r122_tilt(r122_P,r125_gain*r125_tau)
    r125_dnll = np.log(np.maximum(r122_P[np.arange(len(r122_P)),r122_bins],1e-30)) \
                -np.log(np.maximum(r125_Pn[np.arange(len(r122_P)),r122_bins],1e-30))
    print("gain",r125_gain,"CV dNLL",r125_dnll[r122_test].mean(),
          "tau RMS",np.sqrt(np.mean((r125_gain*r125_tau[r122_test])**2)))

print("\n=== TWO-UNIT COMPRESSION ===")
r125_W = np.asarray(load_original("attr_heads.value_bin.2.weight"),dtype=float)
r125_b = np.asarray(load_original("attr_heads.value_bin.2.bias"),dtype=float)
r125_Ffit = r122_F[r122_fit]
r125_G = r125_Ffit.T@r125_Ffit/len(r125_Ffit)+np.eye(193)*1e-6
r125_precision = solve(r125_G,np.eye(193),assume_a="pos")
r125_unique_sd = r122_hsd/np.sqrt(np.diag(r125_precision)[:192])
r125_score = np.linalg.norm(r125_W-r125_W.mean(0),axis=0)*r125_unique_sd
r125_drop = np.argsort(r125_score)[:2]
r125_keep = np.setdiff1d(np.arange(192),r125_drop)
r125_fitH = [r122_H[r122_fit]]
for r125_key,r125_cf in r117_cf.items():
    r125_fitH.append(r125_cf["hidden"][r122_fit & r125_cf["active"]].astype(float))
r125_fitH = np.concatenate(r125_fitH)
r125_kmean = r125_fitH[:,r125_keep].mean(0)
r125_ksd = np.maximum(r125_fitH[:,r125_keep].std(0),.05)
r125_K = np.column_stack(((r125_fitH[:,r125_keep]-r125_kmean)/r125_ksd,
                          np.ones(len(r125_fitH))))
r125_ridge = np.eye(191)*1e-5
r125_ridge[-1,-1]=1e-8
r125_decode = solve(r125_K.T@r125_K/len(r125_K)+r125_ridge,
                    r125_K.T@r125_fitH[:,r125_drop]/len(r125_K),
                    assume_a="pos")
r125_decode_raw = r125_decode[:-1]/r125_ksd[:,None]
r125_decode_b = r125_decode[-1]-r125_kmean@r125_decode_raw
r125_compW = r125_W.copy()
r125_compW[:,r125_keep] += r125_W[:,r125_drop]@r125_decode_raw.T
r125_compW[:,r125_drop] = 0.
r125_compb = r125_b+r125_W[:,r125_drop]@r125_decode_b
r125_deltaL = r122_H@(r125_compW-r125_W).T+(r125_compb-r125_b)
r125_L = np.log(np.maximum(r122_P,1e-30))+r125_deltaL
r125_PC = np.exp(r125_L-logsumexp(r125_L,axis=1)[:,None])
r125_dNC = np.log(np.maximum(r122_P[np.arange(len(r122_P)),r122_bins],1e-30)) \
           -np.log(np.maximum(r125_PC[np.arange(len(r122_P)),r122_bins],1e-30))
print("Dropped units:",r125_drop,"unique scores:",r125_score[r125_drop])
print("Compression CV dNLL:",r125_dNC[r122_test].mean(),
      "centered-logit RMS:",np.std(r125_deltaL[r122_test],axis=1).mean())
print("Active edits:",edited())
```
