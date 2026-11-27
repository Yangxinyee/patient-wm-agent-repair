The recovered kernels imply persistent effects: roughly −1.85 to −0.69 glucose units over six hours per insulin unit, and +3.35 to +0.41 MAP units per norepinephrine rate unit. I’ll compare the 5× edit with matched original states and check the earlier posterior fits. This will distinguish a weak dose signal from a signal the model’s heads fail to use.

```python
r164_s0 = dict(np.load("/work/fwd_4/states_original.npz"))
r164_s5 = dict(np.load("/work/fwd_11/states_edited.npz"))
assert np.array_equal(r164_s0["stay_id"], r164_s5["stay_id"])
assert np.allclose(r164_s0["hour"], r164_s5["hour"])
r164_ids = np.unique(r164_s0["stay_id"])
r164_records = train[train.stay_id.isin(r164_ids)].copy()
r164_h0 = ce_make_heads(r164_records, r164_s0)
r164_h5 = ce_make_heads(r164_records, r164_s5)
print("=== MATCHED HEAD DIFFERENCES ===")
r164_rows = r164_h0["rows"].copy()
for k,h in [("0",r164_h0),("5",r164_h5)]:
    p = np.asarray(h["probs"])
    c = np.asarray(h["centres"])
    b = np.asarray(h["truebin"]).astype(int)
    r164_rows["mean"+k] = (p*c).sum(axis=1)
    r164_rows["nll"+k] = -np.log(np.maximum(p[np.arange(len(b)),b],1e-30))
r164_rows["mean_delta"] = r164_rows.mean5-r164_rows.mean0
r164_rows["nll_delta"] = r164_rows.nll5-r164_rows.nll0
print(r164_rows.groupby("event")[["mean_delta","nll_delta"]].agg(["mean","std"]).round(6).to_string())
print("=== STATE BLOCK CHANGES ===")
r164_dx = r164_s5["x"]-r164_s0["x"]
for label,sl in [("out",slice(0,192)),("slots",slice(192,960)),("posterior",slice(960,992))]:
    a = r164_dx[:,sl]
    print(label, "RMS", round(float(np.sqrt(np.mean(a*a))),6),
          "singular values", np.linalg.svd(a,compute_uv=False)[:6].round(3).tolist())
print("=== RETAINED POSTERIOR FIT OBJECTS ===")
r164_fitkeys = [k for k in globals() if k.startswith(("ds_", "pd_", "r153_", "post_"))
                or ("posterior" in k and not inspect.isfunction(globals()[k]))]
for k in r164_fitkeys[:45]:
    v = globals()[k]
    if isinstance(v,(np.ndarray,torch.Tensor,pd.DataFrame)):
        print(k, type(v).__name__,tuple(v.shape))
    elif isinstance(v,dict):
        print(k,"dict",list(v)[:10])
    elif isinstance(v,(int,float,str)):
        print(k,repr(v)[:150])
print("=== LATE POSTERIOR NOTE ===")
r164_start = r162_notes.find("## Run 153")
print(r162_notes[r164_start:r164_start+1900] if r164_start>=0 else "No run-153 section")
with open("/work/notes.md","a") as f:
    f.write("\n## Run 164\nSix-lag kernels recovered exactly: insulin [-1.850,-2.256,-1.943,-1.561,-1.022,-0.690]; NE [3.351,2.621,1.717,0.943,0.494,0.409]. Constructed matched original/5x heads using exported states, without recurrent replay. Counts: 164 Python runs, 11 forwards.\n")
```