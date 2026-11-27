The saved data include paired states and factual predictions, so we can measure which inputs the local head reconstruction is missing. I’ll check its configuration, register features, and error pattern. A systematic mismatch would require fixing the reconstruction before using it to compute a weight edit.

```python
import inspect
print("--- HELPER SIGNATURES ---")
for nm in ("wlin","wnorm","actfn","wmlp","rc_lin","rc_norm","rc_act","rc_mlp"):
    fn=globals().get(nm)
    if fn is not None:
        print(nm, inspect.signature(fn), "names",fn.__code__.co_names,
              "constants",fn.__code__.co_consts)
print("\n--- BEST REGISTER LAYOUTS ---")
for x in best_layouts[:8]:
    print(str(x)[:700])
print("\n--- RECENT TRIAL RANKING ---")
small_trials=[]
for x in head_trials:
    if isinstance(x,(tuple,list)):
        small_trials.append(str(x)[:180])
    elif isinstance(x,dict):
        small_trials.append(str({k:v for k,v in x.items() if np.isscalar(v) or isinstance(v,tuple)})[:180])
print("\n".join(small_trials[-15:]))
print("\n--- LAST NUMERIC COLUMN NORMS ---")
print(np.linalg.norm(load_original("last_numeric.weight"),axis=0))
for nm in ("ff0","ff1","nf","ctx","zp","cid","lbs"):
    v=globals().get(nm)
    if isinstance(v,(np.ndarray,torch.Tensor)):
        print(nm,tuple(v.shape),v[:2])
print("\n--- RECONSTRUCTION ERROR BY EVENT ---")
if isinstance(best_reconstruction.get("means"),(np.ndarray,torch.Tensor)):
    pm=np.asarray(best_reconstruction["means"])
    if len(pm)==len(head_meta):
        xx=head_meta.assign(local_mean=pm,err=pm-head_meta.tool_mean.values)
        print(xx.groupby("event").agg(n=("err","size"),bias=("err","mean"),
                                     rmse=("err",lambda a:np.sqrt(np.mean(a*a)))))
print("\n--- NUMERICAL HEAD PARAMETERS ---")
for nm in ("gap.0.weight","core.flow.a","z_prior.2.bias","last_status.weight"):
    a0=load_original(nm)
    print(nm, a0.shape, "range",float(a0.min()),float(a0.max()),
          "first",a0.reshape(-1)[:12])
print("\n--- RECORD-BASED IDENTIFICATION NOTES ---")
print(open("/work/notes.md").read()[4200:9000])
print("\n--- HEAD INPUT META SAMPLE ---")
print(hm.head(4).to_string(index=False))
```