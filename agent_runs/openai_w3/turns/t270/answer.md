The fixed FFN features match the installed model, and the cached contrasts are genuine changed-dose feature differences. The variable named `rate154` is not a dose array, so I won’t use it. I’ll extract doses directly from the retained record rows and test the decoder on held-out stays before installing anything.

```python
print("RECORD CONTAINERS", type(r40_info).__name__, len(r40_info),
      "indices", np.asarray(idx81)[:5])
print("FIRST SELECTED RECORD", repr(r40_info[int(np.asarray(idx81)[0])])[:650])
print("\nTWO-COLUMN ARRAY CANDIDATES")
for k242,v242 in dict(globals()).items():
    if isinstance(v242,(np.ndarray,torch.Tensor)) and tuple(v242.shape)==(len(idx81),2):
        a242 = as_numpy(v242) if isinstance(v242,torch.Tensor) else v242
        print(k242, "range", np.round(np.nanmin(a242,axis=0),4),
              np.round(np.nanmax(a242,axis=0),4),
              "first", np.round(a242[:3],4))

def record_rows242(obj):
    if hasattr(obj,"event") and hasattr(obj,"rate"):
        yield obj
    elif isinstance(obj,dict):
        for value in obj.values():
            yield from record_rows242(value)
    elif isinstance(obj,(list,tuple)):
        for value in obj:
            yield from record_rows242(value)

repair_doses242 = np.zeros((len(idx81),2),dtype=np.float64)
repair_rows242 = []
for j242,i242 in enumerate(np.asarray(idx81)):
    for row242 in record_rows242(r40_info[int(i242)]):
        if row242.event not in ("insulin","norepinephrine"):
            continue
        if len(repair_rows242)<4:
            repair_rows242.append(repr(row242))
        rr242 = float(row242.rate)
        if np.isfinite(rr242) and rr242>0:
            repair_doses242[j242,0 if row242.event=="insulin" else 1] += rr242
print("\nEXTRACTED DRUG ROWS", repair_rows242)
print("EXTRACTED DOSE TOTALS", repair_doses242.sum(axis=0),
      "max", repair_doses242.max(axis=0))

if np.all(repair_doses242.sum(axis=0)>0):
    repair_X242 = np.asarray(X220,dtype=np.float64)
    repair_tr242 = np.asarray(tr220,dtype=bool)
    repair_va242 = np.asarray(va220,dtype=bool)
    repair_cov242 = 2.0*(repair_X242[repair_tr242].T @
                         repair_X242[repair_tr242])/repair_tr242.sum()
    repair_rhs242 = np.zeros((769,2))
    repair_targets242 = {}
    for key242,dx242 in contrast_features220.items():
        drug242,mult242 = key242
        yy242 = np.zeros((len(idx81),2))
        yy242[:,drug242] = (mult242-1.0)*repair_doses242[:,drug242]
        yy242 = np.repeat(yy242,4,axis=0)
        repair_targets242[key242] = yy242
        dt242 = np.asarray(dx242,dtype=np.float64)[repair_tr242]
        yt242 = yy242[repair_tr242]
        repair_cov242 += (dt242.T @ dt242)/(len(contrast_features220)*len(dt242))
        repair_rhs242 += (dt242.T @ yt242)/(len(contrast_features220)*len(dt242))
    repair_cov242 += np.eye(769)*1e-4
    repair_beta242 = np.linalg.solve(repair_cov242,repair_rhs242)
    print("\nHELD-OUT FACTUAL SIGNAL RMS",
          np.sqrt(np.mean((repair_X242[repair_va242]@repair_beta242)**2,axis=0)))
    for key242,dx242 in contrast_features220.items():
        pred242 = np.asarray(dx242)[repair_va242]@repair_beta242
        yy242 = repair_targets242[key242][repair_va242]
        drug242 = key242[0]
        nz242 = np.abs(yy242[:,drug242])>1e-8
        slope242 = np.sum(pred242[nz242,drug242]*yy242[nz242,drug242])/np.sum(yy242[nz242,drug242]**2)
        print(key242, "slope", round(float(slope242),3),
              "error RMS", np.round(np.sqrt(np.mean((pred242-yy242)**2,axis=0)),4))
with open("/work/notes.md","a") as f:
    f.write("\nPython run 242; forwards 16. Extracting physical doses directly from retained record rows and fitting a held-out treatment-contrast decoder on fixed FFN features. No weights changed.\n")
```