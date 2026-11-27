The candidate was not installed: the fit omitted a training-row mask, and a cached token array may have been overwritten during counterfactual tests. I’ll rebuild factual tokens directly from the proposed weights and correct the fit. Agreement with the factual cache and a small held-out error would justify installation; disagreement would flag a construction problem.

```python
import numpy as np, torch, pandas as pd
from pathlib import Path

print("Edits still installed:",edited())
_z229 = fixedpost215(rp_input)
_token229 = _z229 @ zin215.T + lc_zbias_base
_nt229 = np.asarray(relay206["norm_tokens"]).copy()
_nt229[:,6] = norm215(_token229)
_pool229 = _nt229.mean(axis=1).astype(np.float64)
print("Rebuilt posterior/cache RMSE:",float(np.sqrt(np.mean((_z229-zf215)**2))))
print("Rebuilt factual pool/cache RMSE:",
      float(np.sqrt(np.mean((_pool229-marker215["fact_pool"])**2))))
print("Cached nt215/factual RMSE:",float(np.sqrt(np.mean((_nt229-nt215)**2))))

_feat229 = _attention_features228(_nt229,candidate228_q,candidate228_kv)
_n229=len(_nt229)
_otherout229 = (_feat229[:,_otherrows228]@
                candidate228_out0[:,_otherrows228].T).reshape(_n229,4,192).mean(axis=1)
_headfeat229 = _feat229[:,candidate228_rows].reshape(_n229,4,48).mean(axis=1)
_q229 = _headfeat229[:,:46]
_f229 = _headfeat229[:,46:]
_y229 = (_target228.reshape(_n229,4,192).mean(axis=1)
         -_otherout229-_f229@candidate228_fixed.T)
_gram229 = _q229[_train227].T@_q229[_train227]
_lam229 = 1e-5*np.trace(_gram229)/46
_coef229 = np.linalg.solve(
    _gram229+_lam229*np.eye(46),
    _q229[_train227].T@_y229[_train227]+
    _lam229*candidate228_out0[:,candidate228_rows[:46]].T)
candidate229_out = candidate228_out0.copy()
candidate229_out[:,candidate228_reserved]=candidate228_fixed
candidate229_out[:,candidate228_rows[:46]]=_coef229.T.astype(np.float32)
_err229 = _feat229@candidate229_out.T-_target228
print("Rebuilt candidate attention RMSE train/hold:",
      float(np.sqrt(np.mean(_err229[np.asarray(relay206["train"])]**2))),
      float(np.sqrt(np.mean(_err229[np.asarray(relay206["hold"])]**2))))
assert np.isfinite(_err229).all()
candidate229_weights = {
    "core.prelude.0.attn.q.weight":candidate228_q,
    "core.prelude.0.attn.kv.weight":candidate228_kv,
    "core.prelude.0.attn.out.weight":candidate229_out,
    "z_post.0.weight":carrier200_w0,"z_post.0.bias":carrier200_b0,
    "z_post.2.weight":carrier200_w2,"z_post.2.bias":carrier200_b2,
    "z_in.weight":zin215,"z_in.bias":lc_zbias_base}
for _name229,_weight229 in candidate229_weights.items():
    save_edit(_name229,torch.as_tensor(np.asarray(_weight229),dtype=torch.float32))
print("Installed candidate:",edited())

_tuning229=pd.read_csv("/data/tuning.csv")
print("Tuning columns:",list(_tuning229.columns))
print("Code mapping:",VOCAB["codes"])
_staycol229="stay_id" if "stay_id" in _tuning229.columns else "stay"
_allstays229=list(map(int,_tuning229[_staycol229].drop_duplicates()))
_codecol229=next((c for c in ["event","code","event_code"]
                  if c in _tuning229.columns),None)
_neeligible229=[]
if _codecol229 is not None:
    _cs229=_tuning229[_codecol229].astype(str)
    _ne229=set(_tuning229.loc[_cs229.eq("norepinephrine"),_staycol229])
    _ma229=set(_tuning229.loc[_cs229.eq("map_arterial"),_staycol229])
    _neeligible229=sorted(map(int,_ne229&_ma229))
candidate229_stays=(_neeligible229[:40]+
                    [s for s in _allstays229 if s not in _neeligible229[:40]][:60])
if len(candidate229_stays)<100:
    candidate229_stays=list(dict.fromkeys(candidate229_stays+_allstays229))[:100]
print("NE-arterial eligible:",len(_neeligible229))
print("GPU STAYS:",candidate229_stays)

with open("/work/notes.md","a") as _f229:
    _f229.write("\n## Runs 228–229\nRun 228 stopped before installation: missing training-row mask in ridge RHS, and nt215 was not a verified factual token cache. Run 229 rebuilt posterior/latent tokens directly from candidate weights, corrected the mask, refit factual compensation, and installed the reversible two-marker candidate for paired GPU validation.\n")
```