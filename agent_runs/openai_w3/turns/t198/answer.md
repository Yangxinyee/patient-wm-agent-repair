The decoder test stopped because the saved posterior weights are NumPy arrays, while the helper requires tensors. No decoding result was produced and no weights changed. I will convert the candidate weights and rerun the test; held-out dose contrasts matching the requested scales would support the carrier approach.

```python
import torch.nn.functional as F
post177 = {k:tensor153(v) for k,v in post154.items()}
z177 = posterior153(ctx164, ev164, uv164, post177)
raw177 = F.linear(z177, tensor153(zin154), tensor153(zb154))
norm177 = F.layer_norm(raw177, (192,),
    load_original("core.prelude.0.norm1.weight"),
    load_original("core.prelude.0.norm1.bias"))
print("Candidate posterior alignment:", float((z177-tensor153(zc164)).square().mean().sqrt()))
print("Stored latent alignment:", float((raw177-tensor153(raw_latent154)).square().mean().sqrt()))
print("Reserved coordinates versus innovations:",
      np.sqrt(np.mean((z177[:,reserved154].numpy()-signal169)**2,axis=0)))
gatevec177 = zero169["core.prelude.0.attn.kv.weight"][338].numpy().astype(np.float64)
features177 = np.column_stack([norm177.numpy().astype(np.float64) @ carrier154,
                               norm177.numpy().astype(np.float64) @ gatevec177])
hold177 = np.isin(stay170, np.unique(stay170)[80:])
fit177 = ~hold177
coef177 = np.linalg.lstsq(features177[fit177], signal169[fit177], rcond=1e-10)[0]
decoder177 = carrier154 @ coef177[:2] + gatevec177[:,None] @ coef177[2:3]
pred177 = norm177.numpy().astype(np.float64) @ decoder177
print("Decoder coefficients:", coef177)
print("Held-out innovation RMS:", np.sqrt(np.mean((pred177[hold177]-signal169[hold177])**2,axis=0)))
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
        ps177 = ns177.numpy().astype(np.float64) @ decoder177
        scenario177[(drug177,factor177)] = ps177
        actual177 = ps177-pred177
        expected177 = np.zeros_like(actual177)
        expected177[:,drug177] = (factor177-1)*Y154[:,drug177]
        sel177 = hold177 & (Y154[:,drug177]>0)
        print("drug",drug177,"factor",factor177,
              "expected/decoded",float(expected177[sel177,drug177].mean()),
              float(actual177[sel177,drug177].mean()),
              "contrast RMS",np.sqrt(np.mean((actual177[sel177]-expected177[sel177])**2,axis=0)))
with open("/work/notes.md","a") as f:
    f.write("\nPython runs 177-178; forwards 10. Run 177 stopped on NumPy/Tensor type mismatch before any edit. Decoder170 was invalid because it used only the constant gate. Fresh actual-carrier decoder tested on held-out dose scaling; no new model weights installed.\n")
```