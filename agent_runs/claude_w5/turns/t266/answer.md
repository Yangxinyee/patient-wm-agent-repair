Precision is ruled out: float32 and float64 local results agree to 5e-7, and both differ from the tool by the same amount at drug bundles. There is a genuine structural difference I haven't identified.

Because z matches exactly, the inputs ctx, e, u and hf are all correct. What remains is how the step uses an action bundle: the memory entry stored for a drug bundle, or the prelude's token set. The doc says memory holds "bundle vectors e", but the tool might store e+u, store u, or add u as an extra entry. A reading bundle right after a drug bundle could still match if its queries hardly attend to that entry.

I'll test these memory variants at the current drug bundle, with a single step from the tool's state at j3, j5, j7 and j9.
- A variant that drops the error to about 1e-6 at all four bundles identifies the convention.
- If none does, the prelude token set is the next suspect.

```python
def msx2(d, mem_mode='e', W=None):
    dd = dict(d)
    e, u, Mv, Ma = d['e'], d['u'], d['Mv'], d['Ma']
    if mem_mode == 'e+u': Mv = torch.cat([Mv[:-1], (e+u)[None]])
    elif mem_mode == 'u': Mv = torch.cat([Mv[:-1], u[None]])
    elif mem_mode == 'extra_u': Mv = torch.cat([Mv, u[None]]); Ma = torch.cat([Ma, Ma[-1:]])
    dd['Mv'], dd['Ma'] = Mv, Ma
    return msx(dd, W=W)
for mode in ['e','e+u','u','extra_u']:
    errs=[]
    for j in [3,5,7,9]:
        dj = prep(j, Xo5); rj = st_vec(msx2(dj, mode)); errs.append(np.linalg.norm(Xo5[idx[j]]-rj)/np.linalg.norm(rj))
    print(mode, np.round(errs,7))
```