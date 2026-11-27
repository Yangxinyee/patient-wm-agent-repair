The candidate posterior and latent vectors now reproduce the saved calculations exactly, but the next comparison stopped on another tensor/array mismatch. I will standardize the diagnostic arrays and finish the held-out scaling test. The carrier is useful only if its decoded changes follow the actual edited rates, not merely the factual innovations.

```python
def np179(x):
    return x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)
target177 = np179(signal169).astype(np.float64)
rates177 = np179(Y154).astype(np.float64)
stays177 = np179(stay170)
print("Reserved coordinates versus innovations:",
      np.sqrt(np.mean((np179(z177[:,reserved154])-target177)**2,axis=0)))
gatevec177 = np179(zero169["core.prelude.0.attn.kv.weight"][338]).astype(np.float64)
features177 = np.column_stack([np179(norm177).astype(np.float64) @ np179(carrier154),
                               np179(norm177).astype(np.float64) @ gatevec177])
hold177 = np.isin(stays177, np.unique(stays177)[80:])
fit177 = ~hold177
coef177 = np.linalg.lstsq(features177[fit177], target177[fit177], rcond=1e-10)[0]
decoder177 = np179(carrier154) @ coef177[:2] + gatevec177[:,None] @ coef177[2:3]
pred177 = np179(norm177).astype(np.float64) @ decoder177
print("Decoder coefficients:", coef177)
print("Held-out innovation RMS:", np.sqrt(np.mean((pred177[hold177]-target177[hold177])**2,axis=0)))
print("Decoder norms:", np.linalg.norm(decoder177,axis=0))
print("Gate mean/SD:", features177[:,2].mean(),features177[:,2].std())
scenario177 = {}
for drug177 in range(2):
    for factor177 in [0.,0.5,1.,1.5,2.,3.]:
        factors177 = [1.,1.]
        factors177[drug177] = factor177
        us177 = action_vectors87(tuple(factors177))
        if isinstance(us177,tuple): us177=us177[0]
        zs177 = posterior153(ctx164,ev164,tensor153(us177),post177)
        rs177 = F.linear(zs177,tensor153(zin154),tensor153(zb154))
        ns177 = F.layer_norm(rs177,(192,),
            load_original("core.prelude.0.norm1.weight"),
            load_original("core.prelude.0.norm1.bias"))
        ps177 = np179(ns177).astype(np.float64) @ decoder177
        scenario177[(drug177,factor177)] = ps177
        actual177 = ps177-pred177
        expected177 = np.zeros_like(actual177)
        expected177[:,drug177] = (factor177-1)*rates177[:,drug177]
        sel177 = hold177 & (rates177[:,drug177]>0)
        print("drug",drug177,"factor",factor177,
              "expected/decoded",float(expected177[sel177,drug177].mean()),
              float(actual177[sel177,drug177].mean()),
              "contrast RMS",np.sqrt(np.mean((actual177[sel177]-expected177[sel177])**2,axis=0)))
with open("/work/notes.md","a") as f:
    f.write("\nPython runs 177-179; forwards 10. Runs 177-178 stopped on type mismatches, without edits; candidate posterior and latent replay nevertheless align exactly. Standardized NumPy diagnostic arrays and tested actual-carrier dose decoding across held-out stays and six scales. Current zero-payload edits unchanged.\n")
```