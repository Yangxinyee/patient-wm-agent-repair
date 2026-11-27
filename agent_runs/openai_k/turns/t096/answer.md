The directions have the intended signs, but the factual-loss calculation failed on a row-selection mask. No candidate was evaluated or submitted. I’ll fix the alignment between readings and bundle states, then rerun the scale comparison. The compression-only candidate should have almost zero loss change; otherwise the local assay itself needs correction.

```python
print("factual alignment:")
for name90 in ["rows80","keep80","lookup80","bins80","mask80","truth0_80"]:
    a90 = np.asarray(globals()[name90])
    print(name90,a90.shape,a90.dtype,
          "range",a90.min(),a90.max(),"first",a90[:3])
mapped90 = np.asarray(lookup80)[np.asarray(rows80,dtype=int)]
valid90 = mapped90 >= 0
if np.asarray(keep80).dtype == bool and len(keep80) == len(valid90):
    valid90 &= np.asarray(keep80)
readrows90 = mapped90[valid90]
readbins90 = np.asarray(bins80)[valid90]
readmask90 = np.asarray(mask80)[valid90]
orig_all90 = np.take_along_axis(probs0_80[readrows90],readbins90[:,:,None],axis=2)[:,:,0]
origp90 = orig_all90[readmask90]
print("aligned reading count:",len(origp90),"original NLL:",
      float(-np.log(np.maximum(origp90,1e-30)).mean()))
if np.asarray(truth0_80).shape == np.asarray(bins80).shape:
    print("saved original probability comparison:",
          float(np.max(np.abs(orig_all90-np.asarray(truth0_80)[valid90]))))

diagnostics90,candidates90 = [],{}
base_mlp90 = mlp0
def hook_mlp90(x,prefix,kind="gelu"):
    if prefix == "z_prior":
        capture74.append(x.detach().clone())
    return base_mlp90(x,prefix,kind)
mlp0 = hook_mlp90
try:
    for scale90 in [0.0,0.25,0.5,1.0,2.0,4.0]:
        weights90 = build_candidate89(scale90)
        W0.update(weights90)
        pf90,mf90,pr90 = dataset80(idx80)
        factp90 = np.take_along_axis(pr90[readrows90],readbins90[:,:,None],axis=2)[:,:,0][readmask90]
        dnll90 = float(np.mean(np.log(np.maximum(origp90,1e-30))-np.log(np.maximum(factp90,1e-30))))
        slopes90 = []
        for c90 in CF89:
            _,cm90,_ = dataset80(c90["indices"],u=c90["u"])
            delta90 = cm90-mf90[lookup80[c90["indices"]]]
            slopes90.append((delta90*c90["dc"][:,None]).sum(0)/(c90["dc"]**2).sum())
        diag90 = {"scale":scale90,"dNLL":dnll90,
                  "factual_mean_RMS":np.sqrt(np.mean((mf90-M0_80)**2,axis=0)).round(4).tolist(),
                  "x2_slopes":np.round(slopes90,3).tolist()}
        diagnostics90.append(diag90)
        candidates90[scale90] = weights90
        print(diag90)
finally:
    W0.update(saved89)
    mlp0 = base_mlp90
print("submitted edits:",edited())
with open("/work/notes.md","a") as f:
    f.write("\n## Run 90: aligned state-token candidate frontier\n"
            "- Fixed factual reading-row mask alignment; tested compression-only and causal scales.\n"
            "- Results retained in diagnostics90/candidates90; all scratch weights restored.\n"
            "- No submitted edits; forward passes remain 4.\n")
```
