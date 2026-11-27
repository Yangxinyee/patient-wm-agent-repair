The fresh context now reconstructs the posterior closely, especially at drug bundles. The raw encoder mismatch may come mainly from empty-action bundles; I’ll check that before using it for dose contrasts. Then I’ll fit a stay-held-out decoder with a factual-zero penalty. Low factual variance and accurate x2 and half-dose contrasts would justify a GPU test.

```python
import numpy as np, torch
from scipy.linalg import solve

rp_has_action=np.bincount(rp_owner.numpy()[rp_action_mask],minlength=len(ds_rates))>0
print("=== ACTION ENCODER CHECK ===")
for _label,_mask in (("action present",rp_has_action),("no action",~rp_has_action)):
    print(_label,"rows",int(_mask.sum()),
          "raw-vs-cache RMS",float(torch.sqrt(torch.mean((rp_raw_u[_mask]-rp_input[_mask,384:])**2))),
          "cached RMS",float(torch.sqrt(torch.mean(rp_input[_mask,384:]**2))))
print("END example:",next((a for a in rp_meta if a.get("end",False)),None))
_active_err=float(torch.sqrt(torch.mean((rp_raw_u[rp_has_action]-rp_input[rp_has_action,384:])**2)))
if _active_err>0.03:
    raise RuntimeError("Active-action encoder mismatch must be resolved before fitting contrasts.")

rp_multipliers=(0.0,0.5,1.5,2.0,3.0)
rp_cf_hidden={}
rp_cf_masks={}
for _j,_drug in enumerate(("insulin","norepinephrine")):
    _indices=[i for i,a in enumerate(rp_meta) if a["code"]==_drug]
    _has=np.zeros(len(ds_rates),dtype=bool)
    _has[rp_owner.numpy()[_indices]]=True
    rp_cf_masks[_drug]=_has
    for _mult in rp_multipliers:
        _atoms=rp_atoms.clone()
        for _i in _indices:
            _a=dict(rp_meta[_i])
            for _field in ("rate","dose"):
                if np.isfinite(_a.get(_field,np.nan)):
                    _a[_field]*=_mult
            _atoms[_i]=rr_atom(**_a)
        _e=rp_encode(_atoms,rp_obs_mask,"encode",rp_encoder_choice[1])
        _u=rp_encode(_atoms,rp_action_mask,"encode_action",True)
        _inp=rp_input.clone()
        _inp[:,192:384]+=_e-rp_raw_e
        _inp[:,384:]+=_u-rp_raw_u
        rp_cf_hidden[(_drug,_mult)]=torch.nn.functional.gelu(_inp@ds_w0.T+ds_b0).numpy()

rp_hold=np.asarray(ds_hold,dtype=bool)&rp_valid
rp_train=rp_valid&~np.asarray(ds_hold,dtype=bool)
rp_H=rp_hidden.numpy().astype(np.float64)
rp_mu=rp_H[rp_train].mean(axis=0)
rp_sd=np.maximum(rp_H[rp_train].std(axis=0),0.05)
rp_A=np.column_stack([(rp_H-rp_mu)/rp_sd,np.ones(len(rp_H))])
rp_D=[]; rp_T=[]; rp_Dhold=[]; rp_Thold=[]
for (_drug,_mult),_h in rp_cf_hidden.items():
    _j=("insulin","norepinephrine").index(_drug)
    _D=np.column_stack([(_h-rp_H)/rp_sd,np.zeros(len(rp_H))])
    _T=np.zeros((len(rp_H),2)); _T[:,_j]=(_mult-1)*ds_rates[:,_j]
    _tr=rp_train&rp_cf_masks[_drug]
    _ho=rp_hold&rp_cf_masks[_drug]
    rp_D.append(_D[_tr]); rp_T.append(_T[_tr])
    rp_Dhold.append(_D[_ho]); rp_Thold.append(_T[_ho])
rp_D=np.vstack(rp_D); rp_T=np.vstack(rp_T)
rp_Dhold=np.vstack(rp_Dhold); rp_Thold=np.vstack(rp_Thold)
rp_Gf=rp_A[rp_train].T@rp_A[rp_train]/rp_train.sum()
rp_Gd=rp_D.T@rp_D/len(rp_D)
rp_R=rp_D.T@rp_T/len(rp_D)
rp_fits={}
print("=== HELD-OUT DOSE-CONTRAST DECODERS ===")
for _pen in (0.1,1.0,3.0,10.0,30.0,100.0):
    _B=solve(rp_Gd+_pen*rp_Gf+1e-4*np.eye(193),rp_R,assume_a="pos")
    _W=_B[:192]/rp_sd[:,None]
    _bias=_B[192]-rp_mu@_W
    rp_fits[_pen]=(_W,_bias)
    _factual=rp_H[rp_hold]@_W+_bias
    _contrast=rp_Dhold@_B
    _ratios=(rp_Thold*_contrast).sum(axis=0)/(rp_Thold**2).sum(axis=0)
    _rmse=np.sqrt(np.mean((_contrast-rp_Thold)**2,axis=0))
    _x2=[]
    for _j,_drug in enumerate(("insulin","norepinephrine")):
        _mask=rp_hold&(ds_rates[:,_j]>0)
        _delta=(rp_cf_hidden[(_drug,2.0)][_mask]-rp_H[_mask])@_W
        _x2.append([float(np.mean(_delta[:,_j]/ds_rates[_mask,_j])),
                    float(np.mean(_delta[:,1-_j]))])
    print("penalty",_pen,
          "factual decoder RMS",np.round(np.sqrt(np.mean(_factual**2,axis=0)),4),
          "contrast slopes",np.round(_ratios,4),
          "contrast RMSE",np.round(_rmse,4),
          "x2 own slope/cross mean",np.round(_x2,4))
print("contrast train/held rows:",len(rp_D),len(rp_Dhold))
with open("/work/notes.md","a") as _f:
    _f.write("\n## Run 174\nBuilt anchored counterfactual posterior features for x0, x0.5, x1.5, x2, x3, and fit closed-form two-output dose-contrast regressions with factual-zero penalties. Holdout is by stay. No new model weights installed.\n")
```