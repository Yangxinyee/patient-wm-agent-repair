Undoing the scale did not resolve the discrepancies, so I won’t use these comparisons to calibrate an edit yet. I’ll audit the probe’s biases and coordinate mapping. A constant offset would be straightforward to remove; a remaining mismatch would mean the probe or local reconstruction needs correction.

```python
print("PROBE PARAMETERS")
for k,v in list(globals().items()):
    if k.startswith("r67"):
        if isinstance(v,(np.ndarray,torch.Tensor)):
            a = as_numpy(v)
            if a.ndim == 1:
                print(k, a.shape, np.round(a[:70],5).tolist())
            else:
                print(k, a.shape)
        elif isinstance(v,(list,tuple)):
            print(k,repr(v)[:450])
        elif isinstance(v,(int,float,str)):
            print(k,repr(v)[:200])
print("\nPAIR BIAS AND MATRIX STRUCTURE")
print("Probe second-layer nonzeros per mean:",
      [(j,np.flatnonzero(r67_W2[j]).tolist(),
        r67_W2[j,np.flatnonzero(r67_W2[j])].tolist()) for j in [0,8,16,24,31]])
print("Probe first-layer rows:",
      [(j,np.flatnonzero(r67_W0[j]).tolist(),
        r67_W0[j,np.flatnonzero(r67_W0[j])].tolist()) for j in [0,8,16,24,31,32,40,56,63]])

def affine_check73(local, observed):
    offset = np.mean(observed-local,axis=0)
    residual = observed-local-offset
    return {"offset":np.round(offset,4).tolist(),
            "residual_mse":float(np.mean(residual**2)),
            "local_sd":np.round(np.std(local,axis=0),3).tolist(),
            "probe_sd":np.round(np.std(observed,axis=0),3).tolist()}
print("\nCONTEXT",affine_check73(cx71.numpy()[:,c71],p72[i71,:8]))
print("OBS",affine_check73(as_numpy(e_all)[:,e71],p72[:,8:24]))
print("ACTION",affine_check73(as_numpy(repair_actual)[:,u71],p72[:,24:32]))
print("No-action probe coordinates:",
      np.round(p72[actn72==0,24:32][:5],4).tolist())
print("\nCONTEXT HELPER BYTECODE")
dis.dis(context_with_flow)
with open("/work/notes.md","a") as f:
    f.write("\nPython run 73; forwards 7. Scale correction alone did not resolve probe comparison. Auditing affine offsets and exact probe tensor construction before trusting its input coordinates. No edits.\n")
```