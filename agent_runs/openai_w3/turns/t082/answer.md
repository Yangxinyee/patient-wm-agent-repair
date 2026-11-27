The probe outputs were deliberately scaled by 0.05, so the raw comparisons overstated the mismatch. I’ll undo that scale and separate errors by bundle type. If context and drug starts then match, we can focus only on the observation and END encoding differences.

```python
p72 = probe7["x"][:,960:992] / np.float32(0.05)
print("SCALE-CORRECTED CONTEXT MSE",
      float(np.mean((cx71.numpy()[:,c71]-p72[i71,:8])**2)))
obsn72 = np.array([sum(r.event not in ("insulin","norepinephrine") for r,end in b)
                  for b in r40_info])
end72 = np.array([any(end for r,end in b) for b in r40_info])
actn72 = np.array([sum(r.event in ("insulin","norepinephrine") for r,end in b)
                  for b in r40_info])
for key, cols, targetcols in [
        ("e_all",e71,slice(8,24)),("e38",e71,slice(8,24)),
        ("e39",e71,slice(8,24)),("repair_actual",u71,slice(24,32)),
        ("u_all",u71,slice(24,32)),("u38",u71,slice(24,32))]:
    a = as_numpy(globals()[key])
    er = np.mean((a[:,cols]-p72[:,targetcols])**2,axis=1)
    print(key, "all",float(er.mean()), "empty obs",float(er[obsn72==0].mean()),
          "obs present",float(er[obsn72>0].mean()),
          "drug starts",float(er[(actn72>0)&~end72].mean()),
          "END",float(er[end72].mean()))
print("\nEMPTY OBSERVATION CANDIDATES")
actual72 = np.median(p72[obsn72==0,8:24],axis=0)
print("Probe median",np.round(actual72,4))
zz72 = torch.zeros(1,192)
sz72 = linear(torch.zeros(1,1),"encode.size")
cand72 = {
    "mlp2(size(0))": layer_norm(mlp(sz72,"encode.mlp2"),"encode.norm"),
    "mlp2(0)": layer_norm(mlp(zz72,"encode.mlp2"),"encode.norm"),
    "norm(size(0))": layer_norm(sz72,"encode.norm"),
    "norm(0)": layer_norm(zz72,"encode.norm")}
for k,v in cand72.items():
    print(k, "MSE",float(np.mean((v.numpy()[0,e71]-actual72)**2)),
          "first coordinates",np.round(v.numpy()[0,e71[:5]],4))
print("\nENCODER BYTECODE")
dis.dis(encode_one_set)
print("\nMLP BYTECODE")
dis.dis(mlp)
with open("/work/notes.md","a") as f:
    f.write("\nPython run 72; forwards 7. Probe mean columns expose 0.05 times selected inputs; scaling was explicitly diagnosed and corrected before interpreting comparisons. Testing context and encoder agreement stratified by observations, drug starts and END bundles. No edits.\n")
```