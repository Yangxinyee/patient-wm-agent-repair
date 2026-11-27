The 5× edit changes hidden slots substantially but barely changes predicted means, especially MAP. Simple dose scaling therefore looks unlikely to fix the main problem. I’ll test whether the posterior exposes a dose-minus-policy signal that can be redirected toward a physiological update; poor held-out decoding would rule out that approach.

```python
import torch.nn.functional as TF
print("=== POSTERIOR RECONSTRUCTION ===")
r165_X = torch.as_tensor(ds_in, dtype=torch.float32)
r165_pre = r165_X @ ds_w0.T + ds_b0
r165_zref = torch.as_tensor(r164_s0["x"][:,960:992])
for label, act in [("gelu",TF.gelu),("silu",TF.silu),("relu",TF.relu)]:
    z = act(r165_pre) @ ds_w2[:32].T + ds_b2[:32]
    print(label, "RMSE", float(torch.sqrt(torch.mean((z-r165_zref)**2))))
print("=== DOSE DECODERS ===")
for key,obj in ds_decoders.items():
    print(key)
    if isinstance(obj,dict):
        for k,v in obj.items():
            if isinstance(v,(np.ndarray,torch.Tensor)):
                print(" ",k,"shape",tuple(v.shape),
                      "norm",round(float(np.linalg.norm(np.asarray(v))),5))
            elif isinstance(v,(int,float,str,list,tuple)):
                print(" ",k,str(v)[:240])
    else:
        print(type(obj).__name__,str(obj)[:180])
print("=== ENCODING HELPERS ===")
for name in ["r86_encode_observations","r89_encode_actions","r79_encode",
             "rr_atom","rr_encode_atoms112","rx_layer2"]:
    fn=globals().get(name)
    if inspect.isfunction(fn):
        print(name,inspect.signature(fn),"dependencies",fn.__code__.co_names)
print("=== FIRST STAY BUNDLES ===")
print(r164_records[r164_records.stay_id==r164_ids[0]]
      [["hour","event","value","rate"]].head(25).to_string(index=False))
print("=== CORE INJECTION TENSORS ===")
print([(n,SHAPES[n]) for n in NAMES
       if any(s in n for s in ["core.b_in","core.c_out","core.prelude_norm",
                               "encode_action.mlp2","encode.mlp2"])])
print("=== INPUT ARRAY ALIGNMENT ===")
print("input shape",ds_in.shape,"state shape",r164_s0["x"].shape,
      "rates mean/std",np.mean(ds_rates,axis=0).round(4),np.std(ds_rates,axis=0).round(4))
with open("/work/notes.md","a") as f:
    f.write("\n## Run 165\n5x matched-state changes: out RMS .03085, slots 1.66381, posterior .07330; approximate head NLL delta +.00309 finger and +.00051 arterial. Large slot changes are mostly not useful output changes. Testing posterior reconstruction and dose-minus-policy decoding for a localized correction. Counts: 165 Python runs, 11 forwards.\n")
```