The caches are aligned, but posterior reconstruction still has a substantial error. Dose decoding is useful for insulin and weaker for norepinephrine. I’ll rebuild the time-flow context directly from exported states and check for a residual-posterior convention. A much smaller error would let us proceed without guessing at the model’s computation.

```python
print("=== HELPER SIGNATURES ===",inspect.signature(norm),inspect.signature(mlp))
print("hour sample:",r164_s0["hour"][:8].tolist())
r167_x=torch.as_tensor(r164_s0["x"],dtype=torch.float32)
r167_same=np.r_[False,r164_s0["stay_id"][1:]==r164_s0["stay_id"][:-1]]
r167_prev=torch.cat([torch.zeros_like(r167_x[:1]),r167_x[:-1]],dim=0)
r167_prev[~torch.as_tensor(r167_same)]=0
r167_dt=np.r_[1.,np.diff(r164_s0["hour"])*3600.]
r167_dt[~r167_same]=1.
r167_dt=torch.as_tensor(np.maximum(r167_dt,1.),dtype=torch.float32)
r167_flow=torch.exp(-r167_dt[:,None]*torch.exp(
    torch.as_tensor(load_original("core.flow.a")))[None,:])
r167_hflow=r167_prev[:,192:960].reshape(-1,4,192)*r167_flow[:,None,:]
r167_base=r167_prev[:,:192]+norm(r167_hflow.mean(1),"core.out_norm")
r167_results=[]
r167_contexts={}
for name,gg in [("log/10",torch.log(r167_dt)/10),
                ("log1p/10",torch.log1p(r167_dt)/10),
                ("log",torch.log(r167_dt)),
                ("log1p",torch.log1p(r167_dt))]:
    cc=r167_base+mlp(gg[:,None],"gap","silu")
    r167_contexts[name]=cc
    inp=torch.cat([cc,r165_X[:,192:]],dim=1)
    pp=TF.gelu(inp@ds_w0.T+ds_b0)@ds_w2.T+ds_b2
    prior=mlp(cc,"z_prior","gelu")[:,:32]
    mask=torch.as_tensor(r167_same)
    for convention,zz in [("raw",pp[:,:32]),("plus_prior",pp[:,:32]+prior),
                          ("tanh",torch.tanh(pp[:,:32]))]:
        er=float(torch.sqrt(torch.mean((zz[mask]-r165_zref[mask])**2)))
        r167_results.append((er,name,convention))
    print(name,"context-cache RMSE",
          float(torch.sqrt(torch.mean((cc[mask]-r165_X[mask,:192])**2))))
print("posterior tests:",sorted(r167_results))
print("=== RR ENCODING OBJECTS ===")
for k in [k for k in globals() if k.startswith("rr_") and
          not inspect.isfunction(globals()[k])][:45]:
    v=globals()[k]
    print(k,tuple(v.shape) if hasattr(v,"shape") else
          (list(v)[:5] if isinstance(v,dict) else type(v).__name__))
print("=== ALL CORE PARAMETER NAMES ===")
print([(n,SHAPES[n]) for n in NAMES if n.startswith("core.")])
with open("/work/notes.md","a") as f:
    f.write("\n## Run 167\nAligned caches confirmed. Held-out active-dose RMSE: insulin .280 from action and .575 from context; NE .217 from action and .269 from context. Rebuilding posterior context from exported previous states to resolve remaining reconstruction error. Counts: 167 Python runs, 11 forwards.\n")
```