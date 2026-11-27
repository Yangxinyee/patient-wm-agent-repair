The probe retained an output bias. After accounting for it, the context reconstruction appears exact, so that part of the diagnostic is sound. I’ll subtract the bias explicitly and check which bundle encodings still disagree. This should distinguish genuine encoder differences from probe bookkeeping errors.

```python
probe_inputs74 = (probe7["x"][:,960:992]-r67_probe_bias[None,:])/r67_probe_scale
print("CONTEXT MSE",float(np.mean(
    (cx71.numpy()[:,c71]-probe_inputs74[i71,:8])**2)))
groups74 = {
    "observations":obsn72>0,
    "single WORLD drug":(obsn72==0)&(actn72==1)&~end72,
    "two WORLD drugs":(obsn72==0)&(actn72==2)&~end72,
    "END bundles":end72,
    "no actions":actn72==0}
for key, cols, targetcols in [
        ("e_all",e71,slice(8,24)),("e39",e71,slice(8,24)),
        ("repair_actual",u71,slice(24,32)),("u_all",u71,slice(24,32)),
        ("u38",u71,slice(24,32))]:
    a = as_numpy(globals()[key])
    er = np.mean((a[:,cols]-probe_inputs74[:,targetcols])**2,axis=1)
    print(key, {g:(int(m.sum()),round(float(er[m].mean()),9))
                for g,m in groups74.items() if m.any()})
print("\nEMPTY ACTION")
a74 = probe_inputs74[actn72==0,24:32]
print("Actual mean",np.round(a74.mean(0),6).tolist(),
      "SD",np.round(a74.std(0),6).tolist())
print("Action norm bias",np.round(as_numpy(W["encode_action.norm.bias"])[u71],6).tolist())
print("\nEMPTY OBSERVATION")
for g in ["single WORLD drug","two WORLD drugs","END bundles"]:
    m=groups74[g]& (obsn72==0)
    if m.any():
        print(g,"mean",np.round(probe_inputs74[m,8:24].mean(0),5).tolist(),
              "SD",np.round(probe_inputs74[m,8:24].std(0),5).tolist())
for k,v in cand72.items():
    print(k,"single-drug MSE",float(np.mean(
        (v.numpy()[0,e71]-probe_inputs74[groups74["single WORLD drug"],8:24])**2)))
print("\nENCODER PARAMETERS",[(n,SHAPES[n]) for n in NAMES
                             if n.startswith(("encode.","encode_action."))])
with open("/work/notes.md","a") as f:
    f.write("\nPython run 74; forwards 7. Identified retained probe output bias. Exact exposed inputs are (saved z_mean - r67_probe_bias)/0.05; context residual after removing its affine offset was approximately 1e-8. Comparing correctly decoded e/u by bundle type, including normalized-empty action baseline. No edits.\n")
```