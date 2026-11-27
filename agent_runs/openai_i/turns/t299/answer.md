The GPU states reveal the failure: before treatment, the supposed norepinephrine innovation channel is around 5, not zero, and the slot-state RMS change is 14.4. That refutes the local gating result. I’ve restored the neutral edit. I’ll now compare the saved posterior calculation with the GPU behavior; a repair must make the private channels effectively zero on drug-free bundles.

```python
import torch.nn.functional as F
print("PRIVATE INDICES", private_indices209, "DIMS", private_dims196)
inp264 = torch.as_tensor(array196(input196), dtype=torch.float32)
mask264 = np.asarray(inactive209, bool)
print("INPUT SHAPE", tuple(inp264.shape),
      "INACTIVE ACTION NORM", np.max(np.linalg.norm(array196(inp264[mask264,384:]),axis=1)))
archives264 = {}
for fn264 in ("neutral209.npz","small_route222.npz","sharp243.npz"):
    zz264 = np.load("/work/"+fn264)
    archives264[fn264] = {k:zz264[k] for k in zz264.files}
    w0264=torch.as_tensor(zz264["z_post.0.weight"],dtype=torch.float32)
    b0264=torch.as_tensor(zz264["z_post.0.bias"],dtype=torch.float32)
    w2264=torch.as_tensor(zz264["z_post.2.weight"],dtype=torch.float32)
    b2264=torch.as_tensor(zz264["z_post.2.bias"],dtype=torch.float32)
    pre264=F.linear(inp264,w0264,b0264)
    print(fn264,"POSTERIOR PRIVATE ROW NORMS",
          torch.linalg.vector_norm(w2264[[16,12]],dim=1).tolist(),
          "BIASES",b2264[[16,12]].tolist())
    for approx264 in ("none","tanh"):
        mu264=array196(F.linear(F.gelu(pre264,approximate=approx264),w2264,b2264)[:,:32])
        print(" ",approx264,"inactive RMS",
              np.sqrt(np.mean(mu264[mask264][:,[16,12]]**2,axis=0)).tolist(),
              "inactive minmax",
              [(float(mu264[mask264,j].min()),float(mu264[mask264,j].max())) for j in (16,12)])
    print(" neutral differences",
          {k:round(float(np.linalg.norm(zz264[k]-archives264["neutral209.npz"][k])),6)
           for k in archives264["neutral209.npz"]})
print("=== RECONSTRUCTION CONTRACTS ===")
for nm264 in ("encode_rows152","posterior137","atom137","pool137",
              "replay_inputs161","core_step136"):
    fn264=globals()[nm264]
    print(nm264,inspect.signature(fn264),"names",fn264.__code__.co_names,
          "constants",str(fn264.__code__.co_consts)[:180])
print("=== CACHED METADATA ===")
for nm264 in ("meta196","all_meta196","cal_meta161","meta182","meta201","ctx196","input196",
              "action196","u196","all_z201","original_z216"):
    if nm264 in globals():
        vv264=globals()[nm264]
        print(nm264,type(vv264).__name__,np.shape(vv264),
              list(vv264.columns) if isinstance(vv264,pd.DataFrame) else
              list(vv264.keys())[:8] if isinstance(vv264,dict) else "")
with open("/work/notes.md","a") as f:
    f.write("\n## Python 264 — GPU inactive-channel contradiction\n"
            "- Sharp243 GPU private NE coordinate is ~5 on drug-free bundles, despite local zero-innovation claims.\n"
            "- Before treatment: out RMS .2551, slot RMS 14.3815. This explains why attention responses damaged factual fit.\n"
            "- Neutral209 restored; investigating posterior/input or activation mismatch before any new response edit.\n")
```
