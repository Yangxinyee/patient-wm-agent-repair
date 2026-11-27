The recovered estimates give us quantitative targets. The remaining reconstruction uncertainty is mainly around observations and infusion endings. I’ll compare the probe’s inputs with the local encoders; an exact match would let us calibrate weight changes locally and reserve forward passes for validation.

```python
p71 = probe7["x"][:,960:992]
c71 = np.array([0,27,54,81,109,136,163,191])
e71 = np.array([0,12,25,38,50,63,76,89,101,114,127,140,152,165,178,191])
u71 = c71.copy()
print("PROBE WEIGHT CHECK")
for key in ["r67_W0","r67_W2"]:
    a = globals().get(key)
    if a is not None:
        a = as_numpy(a)
        print(key, [(i, np.flatnonzero(a[i]).tolist()[:5],
                     a[i,np.flatnonzero(a[i])].tolist()[:5]) for i in range(4)])
print("STATE KEYS", [(k,tuple(v.shape)) for k,v in S.items()])
print("BUNDLE INFO", type(r40_info).__name__, len(r40_info),
      repr(r40_info[:3])[:1400])
print("PROBE/ORIGINAL ALIGNMENT",
      np.array_equal(probe7["stay_id"],S["stay_id"]),
      np.max(np.abs(probe7["hour"]-S["hour"])))
print("\nOBSERVATION ENCODER COMPARISONS")
for key in ["e_all","e38","e39"]:
    a = as_numpy(globals()[key])
    print(key, "MSE",float(np.mean((a[:,e71]-p71[:,8:24])**2)))
print("Actual empty-observation coordinates, median:",
      np.round(np.median(p71[np.sum(r41_rates,axis=1)>0,8:24],axis=0),4))
for key in ["e_empty","empty_obs","clean_empty_obs"]:
    a = as_numpy(globals()[key])
    print(key, np.round(a[0,e71],4))
print("\nACTION ENCODER COMPARISONS")
for key in ["u_all","u38","u39","u39_zeroed","repair_actual","r48_U"]:
    a = as_numpy(globals()[key])
    print(key, "MSE",float(np.mean((a[:,u71]-p71[:,24:32])**2)))
print("\nPROBE CONTEXT RECONSTRUCTION")
i71 = np.flatnonzero(probe7["stay_id"][1:]==probe7["stay_id"][:-1])+1
xp71 = torch.from_numpy(probe7["x"]).float()
dt71 = torch.from_numpy((probe7["hour"][i71]-probe7["hour"][i71-1])*3600).float()
hf71 = xp71[i71-1,192:960].reshape(-1,4,192)*torch.exp(
    -dt71[:,None,None]*torch.exp(W["core.flow.a"])[None,None,:])
cx71 = xp71[i71-1,:192]+layer_norm(hf71.mean(1),"core.out_norm")+mlp(
    (torch.log1p(dt71)/10)[:,None],"gap")
print("Context MSE:",float(np.mean((cx71.numpy()[:,c71]-p71[i71,:8])**2)))
print("Observed encoder norm:",np.round(np.mean(p71[:,8:24]**2),5),
      "action encoder norm:",np.round(np.mean(p71[:,24:32]**2),5))
with open("/work/notes.md","a") as f:
    f.write("\nPython run 71; forwards 7. Comparing exact probe posterior input coordinates against retained local encodings and context reconstruction, with original weights restored. This diagnoses preprocessing rather than attempting another ineffective latent edit.\n")
```