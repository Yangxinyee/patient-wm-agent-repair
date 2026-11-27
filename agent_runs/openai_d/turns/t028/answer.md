The original weights are restored. The architecture confirms that doses enter a separate action pathway, but its register description does not fully explain the reconstruction attempts. I’ll inspect the saved states and head configuration before choosing an edit; this should show whether the remaining mismatch is in state alignment or head inputs.

```python
print("--- HEAD TENSOR SHAPES ---")
for name in NAMES:
    if (name.startswith(("last_", "attr_heads.value", "z_head", "z_prior",
                         "query_norm", "type_emb", "gap", "embed_transition",
                         "encode_action", "encode.", "core.z_", "core.gap"))
        or name in ("embed.numeric.weight", "embed.numeric.bias")):
        print(name, SHAPES[name])
print("\n--- FUNCTIONS ---")
print([k for k,v in list(globals().items())
       if callable(v) and getattr(v,"__module__","")=="__main__"])
print("\n--- STATE VARIABLES ---")
for k,v in list(globals().items()):
    if any(s in k.lower() for s in ("state","meta","reconstruct","sample","norm")):
        if isinstance(v, dict):
            print(k, "dict keys", list(v.keys())[:12])
        elif isinstance(v, (np.ndarray, torch.Tensor, pd.DataFrame)):
            print(k, type(v).__name__, v.shape)
print("\n--- CONFIGURATIONS ---")
for k in ("cfg","config","info0","best_reconstruction","check","candidate"):
    v=globals().get(k)
    if isinstance(v,dict):
        print(k,{a:b for a,b in v.items()
                 if a in ("cfg","config","rmse","noise_rms_4draw","bias","expected_four_draw_rmse")})
    elif v is not None:
        print(k,v)
print("\n--- HEAD MATCHED ROWS ---")
print(head_meta.head(8).to_string(index=False))
print("\n--- VOCAB ---")
for k,v in VOCAB.items():
    if k not in ("value_bin_centres",):
        ss=str(v)
        print(k, ss[:2300])
print("\n--- STATE FILE CONTENTS ---")
for d in ("/work/fwd_1","/work/fwd_2","/work/fwd_3","/work/fwd_4"):
    print(d,os.listdir(d))
    for fn in os.listdir(d):
        if fn.endswith(".npz"):
            npz=np.load(os.path.join(d,fn))
            print(fn,{k:npz[k].shape for k in npz.files})
print("\n--- EARLY NOTEBOOK ---")
print(open("/work/notes.md").read()[:4600])
```