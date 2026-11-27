The reconstruction check failed before the tuning cache was built, so no weights changed. The mismatch could come from the recovered formula or from an old cache left by a probe. I’ll compare context, prior, query, and register terms separately; a constant offset would point to the register, while variation across latent draws would point elsewhere.

```python
print("=== RECONSTRUCTION COMPONENT CHECKS ===")
r202_refs = [
    ("ctx",r201_tc,r124_ctx[r201_sel]),
    ("prior",r201_mlp(r201_tc,"z_prior"),r124_pp[r201_sel]),
    ("query",r201_tq,r129_q_o[:,r201_sel]),
]
for label,a,b in r202_refs:
    d=a-b
    print(label,"max",float(d.abs().max()),"rms",float(d.square().mean().sqrt()))
r202_f = r201_fixed(r201_tm,True,True)[0]
r202_d = r124_x[:,r201_sel] - r201_tq - r202_f[None]
print("full-input discrepancy RMS",float(r202_d.square().mean().sqrt()),
      "sample-dependent RMS",float((r202_d-r202_d.mean(0,keepdim=True)).square().mean().sqrt()))
print("discrepancy by first rows",r202_d.mean(0).square().mean(1).sqrt()[:12].tolist())
print("=== r124_W DIFFERENCES FROM ORIGINAL ===")
r202_diffs=[]
for n,w in r124_W.items():
    if n in r201_W and tuple(w.shape)==tuple(r201_W[n].shape):
        delta=float((w-r201_W[n]).abs().max())
        if delta>1e-6:
            r202_diffs.append((n,delta))
print(r202_diffs[:30])
print("=== HEAD-CACHE VARIABLE REFERENCES ===")
r202_names=[]
for k in sorted(list(globals())):
    v=globals()[k]
    if isinstance(v,(torch.Tensor,np.ndarray)) and len(v.shape)>=2:
        if tuple(v.shape[-2:]) in [(4510,192),(4510,64),(4510,17)] or tuple(v.shape)==(4510,192):
            r202_names.append((k,tuple(v.shape)))
print(r202_names[:70])
print("=== ORIGINAL REGISTER VALUES ===")
print(r201_tm[["event","has_last","age","last_z","last_bin"]].head(8).to_string(index=False))
print("last_status/absent-bin norms",
      float(r201_W["last_status.weight"][0].norm()),
      float(r201_W["embed.bins.rate.weight"][0].norm()),
      float(r201_W["embed.bins.value.weight"][0].norm()))
for name in ["rh_rcfg","r98_cfg","r98_best","r99_cfg","r99_best"]:
    if name in globals():
        print(name,repr(globals()[name])[:500])
with open("/work/notes.md","a") as f:
    f.write("\n## Run 201–202: reconstruction assertion failed safely\n"
            "No edits changed. The r124_x comparison failed by 2.218; "
            "auditing old-cache provenance and component differences before using local losses.\n")
```
