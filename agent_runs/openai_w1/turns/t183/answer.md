The cached posterior reconstruction did not match the exported posterior means, so it is not safe to use it for an edit yet. I’ll check input and output ordering, cache alignment, and held-out dose decoding. A close reconstruction would validate the pathway; a persistent mismatch would require a different approach.

```python
from sklearn.metrics import r2_score, mean_squared_error
print("=== CACHE ALIGNMENT ===")
for key in ["x","stay_id","hour"]:
    a=np.asarray(ds_s[key]); b=np.asarray(r164_s0[key])
    print(key,a.shape,b.shape,
          float(np.max(np.abs(a-b))) if a.shape==b.shape else "different shapes")
print("weights original:",
      torch.allclose(ds_w0,torch.as_tensor(load_original("z_post.0.weight"))),
      torch.allclose(ds_w2,torch.as_tensor(load_original("z_post.2.weight"))))
print("posterior reference mean/std:",r165_zref.mean().item(),r165_zref.std().item())
import itertools
r166_tests=[]
for order in itertools.permutations(range(3)):
    xx=torch.cat([r165_X[:,j*192:(j+1)*192] for j in order],dim=1)
    pp=xx@ds_w0.T+ds_b0
    for label,act in [("gelu",TF.gelu),("silu",TF.silu),("relu",TF.relu)]:
        zz=act(pp)@ds_w2.T+ds_b2
        for layout,zzm in [("split",zz[:,:32]),("interleaved",zz[:,::2])]:
            er=float(torch.sqrt(torch.mean((zzm-r165_zref)**2)))
            r166_tests.append((er,order,label,layout))
print("best reconstructions:",sorted(r166_tests)[:6])
print("=== HELD-OUT DOSE DECODING ===")
r166_hold=np.asarray(ds_hold).astype(bool)
if r166_hold.shape!=(len(ds_in),):
    r166_hold=np.isin(r164_s0["stay_id"],r164_ids[::5])
for key,reg in ds_decoders.items():
    drug,source=key
    xx={"action":ds_in[:,384:576],
        "context":ds_in[:,:192],
        "context_observation":ds_in[:,:384]}[source]
    yy=ds_rates[:,0 if drug=="insulin" else 1]
    pred=reg.predict(xx)
    flag=np.asarray(r164_s0["has_"+drug]).astype(bool)
    for label,mask in [("all",r166_hold),("drug",r166_hold&flag)]:
        if mask.sum()>1:
            print(key,label,"n",int(mask.sum()),"R2",round(r2_score(yy[mask],pred[mask]),4),
                  "RMSE",round(float(np.sqrt(np.mean((yy[mask]-pred[mask])**2))),4))
print("=== ENCODING CACHE SIZES ===")
for k in ["r36_ctx","r71_e_all","r71_u_all","r86_atoms","r89_atoms",
          "rr_atoms4","rr_owner4","rr_original","W"]:
    v=globals().get(k)
    print(k,tuple(v.shape) if hasattr(v,"shape") else
          (list(v)[:4] if isinstance(v,dict) else type(v).__name__))
with open("/work/notes.md","a") as f:
    f.write("\n## Run 166\nPosterior reconstruction from ds_in failed (best tested initially RMSE .218), so no posterior edit yet. Checking cache alignment, input/output layout, and held-out dose decoders before trusting this pathway. Counts: 166 Python runs, 11 forwards.\n")
```