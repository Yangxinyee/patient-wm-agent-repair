The fresh step matches the saved states. Median relative error is about 1e-7 for out and h and 3e-7 for z; the worst step is 6e-3 on z. So I have an exact local model of the core and can compute states for any stay without forward passes.

Next I record this in the notebook and check a free run from the initial state over a whole stay. I try `h_init`/`o_init` and a few first-step time gaps and keep whichever reproduces the saved first-bundle state. A match means I can run the local model on any training stay.

```python
with open('/work/notes.md','a') as f:
    f.write("""
## Runs 119-121: FRESH local implementation (float64) verified teacher-forced: rel err median 1e-7 (out,h), 3e-7 (z), max 6e-3.
- kernel fns: avec(atom), senc(vecs,name), enc2(atoms)->(e,u), cstep(out,h,dt_s,e,u,Mv,Mage)->(out,y,z,E,hf), tf_check(sid).
- memory = e of last <=128 bundles incl current, ages in seconds; z-score clip +-10; fine bin = 1 + #edges < x.
- states npz keys: x, stay_id, hour, has_insulin, has_norepinephrine, n_atoms.
""")
def free_run(B, h0, o0, dt0):
    es=[]; outs=[]; ys=[]; zs=[]; Es=[]
    out,h = o0.clone(), h0.clone()
    for k,(t,atoms) in enumerate(B):
        e,u = enc2(atoms); es.append(e); j0=max(0,k-127)
        dt_s = dt0 if k==0 else (t-B[k-1][0])*3600
        Mv=torch.stack(es[j0:k+1]); Mage=torch.tensor([(t-B[j][0])*3600 for j in range(j0,k+1)],dtype=torch.float64)
        out,h,z,E,_ = cstep(out,h,dt_s,e,u,Mv,Mage); outs.append(out); ys.append(h); zs.append(z); Es.append(E)
    return torch.stack(outs), torch.stack(ys), torch.stack(zs), torch.stack(Es)
rows = np.where(SIDS==sidA)[0]; Xt_ = torch.tensor(X_[rows],dtype=torch.float64)
for h0n,o0n in [('h_init','o_init'),('zero','zero')]:
    h0 = h_init.double() if h0n=='h_init' else torch.zeros(4,192,dtype=torch.float64)
    o0 = o_init.double() if o0n=='o_init' else torch.zeros(192,dtype=torch.float64)
    for dt0 in [1.0, 60.0, 3600.0]:
        O,Y,Z,_ = free_run(bundles_B[sidA], h0, o0, dt0)
        r0 = float((Y[0].reshape(-1)-Xt_[0,192:960]).norm()/Xt_[0,192:960].norm()); rl = float((Y[-1].reshape(-1)-Xt_[-1,192:960]).norm()/Xt_[-1,192:960].norm())
        print(h0n, dt0, 'first-row h err %.2e  last-row h err %.2e' % (r0, rl))
```