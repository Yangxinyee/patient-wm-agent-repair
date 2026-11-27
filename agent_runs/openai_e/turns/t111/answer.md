Correcting the inputs exposes much more factual activation in the old NE detector, and its actual halving sensitivity exceeds its doubling sensitivity. That helps explain the failed trial. I’ll now check whether original factual MAP errors retain a dose-innovation signal. If they do, a correction centered on that missing signal may improve factual fit rather than merely preserve it.

```python
import numpy as np, pandas as pd, torch, inspect
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from pathlib import Path

fact96 = pd.read_csv("/work/fwd_2/factual_original.csv")
map96 = fact96[fact96.event.eq("map_arterial")].copy()
map96["hour"] = np.round(map96["hour"].astype(float)-1.0,6)
map96["error"] = map96["observed"]-map96["predicted_mean"]
join96 = cf_frame[["stay_id","hour"]].copy()
join96["hour"] = np.round(join96["hour"].astype(float),6)
join96["row96"] = np.arange(len(join96))
join96 = join96.merge(
    map96[["stay_id","hour","observed","predicted_mean","error"]],
    on=["stay_id","hour"],how="left",validate="one_to_one")
map_error96 = join96.sort_values("row96")["error"].to_numpy()
map_mean96 = join96.sort_values("row96")["predicted_mean"].to_numpy()
map_observed96 = join96.sort_values("row96")["observed"].to_numpy()
dose_ne96 = np.asarray(cf_doses,dtype=float)[:,1]
ne_present96 = dose_ne96>0
groups96 = cf_frame["stay_id"].to_numpy()

ctx96 = cf_C.numpy().astype(float)
prior_hidden96 = F.gelu(F.linear(
    cf_C,W0["z_prior.0.weight"],W0["z_prior.0.bias"])).numpy().astype(float)
Xctx96 = np.column_stack([ctx96,prior_hidden96])
ne_dose_pred96 = np.full(len(cf_frame),np.nan)
ne_error_pred96 = np.full(len(cf_frame),np.nan)
indices96 = np.flatnonzero(ne_present96)
folds96 = GroupKFold(n_splits=5)
for fitpos,testpos in folds96.split(
        Xctx96[indices96],dose_ne96[indices96],groups96[indices96]):
    fitidx,testidx = indices96[fitpos],indices96[testpos]
    center = Xctx96[fitidx].mean(axis=0)
    scale = np.maximum(Xctx96[fitidx].std(axis=0),.05)
    model = Ridge(alpha=40.).fit(
        (Xctx96[fitidx]-center)/scale,dose_ne96[fitidx])
    ne_dose_pred96[testidx] = model.predict((Xctx96[testidx]-center)/scale)
    fiterror = fitidx[np.isfinite(map_error96[fitidx])]
    if len(fiterror)>30:
        emodel = Ridge(alpha=100.).fit(
            (Xctx96[fiterror]-center)/scale,map_error96[fiterror])
        ne_error_pred96[testidx] = emodel.predict((Xctx96[testidx]-center)/scale)

innovation96 = dose_ne96-ne_dose_pred96
mask96 = ne_present96 & np.isfinite(map_error96) & np.isfinite(ne_error_pred96)
dr96 = innovation96[mask96]
er96 = (map_error96-ne_error_pred96)[mask96]
beta96 = float(dr96@er96/(dr96@dr96))
score96 = dr96*(er96-beta96*dr96)
cluster96 = pd.Series(score96).groupby(groups96[mask96]).sum().to_numpy()
se96 = float(np.sqrt(np.sum(cluster96**2))/(dr96@dr96))
print("FACTUAL DOSE-INNOVATION AUDIT")
print("Drug bundles",int(ne_present96.sum()),"next arterial readings",int(mask96.sum()))
print("Dose SD",float(np.std(dose_ne96[ne_present96])),
      "cross-fitted innovation SD",float(np.std(innovation96[ne_present96])))
print("Original next-MAP error mean/SD",
      float(np.mean(map_error96[mask96])),float(np.std(map_error96[mask96])))
print("Residual error per dose innovation",beta96,"cluster SE",se96)
print("Dose predictor R2",float(1-
    np.mean(innovation96[ne_present96]**2)/
    np.var(dose_ne96[ne_present96])))

print("\nCORE DIAGNOSTIC INPUTS")
for name,obj in sorted(list(globals().items())):
    if isinstance(obj,(np.ndarray,torch.Tensor)) and (
        "82" in name or any(s in name for s in [
            "hnew","h_update","head_input","input_head","core_h_flow",
            "core_U_actions","core_E_obs","core_z","head_index"])):
        print(name,tuple(obj.shape))
    elif isinstance(obj,dict) and ("config" in name.lower() or name=="core_memories"):
        print(name,"keys",list(obj)[:12])
        if "config" in name.lower() and len(obj)<18:
            print(str(obj)[:800])
print("\nCORE UPDATE CONTRACT")
print(inspect.signature(core_update_local))
print("locals",core_update_local.__code__.co_varnames)
print("constants",core_update_local.__code__.co_consts)
print("head input candidates")
for name,obj in sorted(list(globals().items())):
    if isinstance(obj,(np.ndarray,torch.Tensor)) and obj.ndim==2 and obj.shape[0]==4751:
        print(name,tuple(obj.shape))

with open("/work/notes.md","a") as f:
    f.write(
        "\n## Run 96: factual dose-innovation error audit\n"
        "Corrected old NE detector has materially greater factual leakage and "
        "halving than doubling sensitivity. Tested whether original next-hour "
        "arterial MAP residuals retain a cross-fitted dose-innovation signal, "
        "using only tuning records and original predictions. This is a diagnostic, "
        "not an independent causal estimate. No weights changed. "
        f"Audit beta={beta96:.4f}, cluster SE={se96:.4f}, n={int(mask96.sum())}. "
        "Python runs: 96; forwards: 12.\n")
```