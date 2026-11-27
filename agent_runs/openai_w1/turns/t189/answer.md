The cached posterior uses GELU and stores its mean in the first 32 outputs. I also found the newer, more accurate input cache. I’ll use that cache to test dose contrasts while constraining the correction to stay small on factual records. If those contrasts remain decodable on held-out stays, a low-rank posterior edit is worth testing.

```python
import numpy as np, torch

print("=== FRESH POSTERIOR VALIDATION ===")
_p = (r168_hidden @ ds_w2.T + ds_b2)[:, :32].numpy()
_m = np.asarray(r168_valid, dtype=bool)
print("RMSE all / valid:",
      float(np.sqrt(np.mean((_p-r168_zactual)**2))),
      float(np.sqrt(np.mean((_p[_m]-r168_zactual[_m])**2))))
print("valid bundles:", int(_m.sum()))
print("hidden consistency:",
      float(torch.max(torch.abs(r168_hidden-torch.nn.functional.gelu(r168_in@ds_w0.T+ds_b0)))))
print("fresh-vs-old input RMS by block:",
      [float(torch.sqrt(torch.mean((r168_in[:,192*i:192*(i+1)]-torch.as_tensor(ds_in[:,192*i:192*(i+1)]))**2))) for i in range(3)])
print("=== REUSABLE EMBEDDING HELPERS ===")
for _fn in (rr_atom, rr_encode_atoms112, rr_linear, rr_mlp, rr_norm):
    print(_fn.__name__, "defaults:", _fn.__defaults__,
          "dependencies:", _fn.__code__.co_names,
          "constants:", [repr(_c)[:100] for _c in _fn.__code__.co_consts if isinstance(_c,(str,int,float,tuple))][:18])
print("=== ALIGNED DRUG BUNDLE EXAMPLES ===")
for _j,_drug in enumerate(("insulin","norepinephrine")):
    _i = int(np.flatnonzero(_m & (ds_rates[:,_j]>0))[0])
    _sid = int(ds_s["stay_id"][_i]); _hour=float(ds_s["hour"][_i])
    _key=(_sid,int(round(_hour*60)))
    print(_drug,"row",_i,"stay/hour",_sid,_hour,"rates",ds_rates[_i],
          "metadata",repr(rr_bundle_map4.get(_key))[:1700])
print("=== DRUG VOCABULARY ===")
print("numeric:", {str(_k):_v for _k,_v in rr_numeric.items() if _k[0] in ("insulin","norepinephrine")})
print("code ids:", {k:rr_codes[k] for k in ("insulin","norepinephrine")})
print("fresh action matches rr_u4 RMS:",float(torch.sqrt(torch.mean((r168_in[:,384:]-rr_u4)**2))))
with open("/work/notes.md","a") as _f:
    _f.write("\n## Run 172\nUsing r168_in/r168_hidden rather than stale ds_in. Plan: fit dose-contrast decoders with explicit factual-zero penalties, then inject record-supported physical latent directions through z_post.2 only. Candidate remains unvalidated.\n")
```