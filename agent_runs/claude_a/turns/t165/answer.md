`P`, `PO`, `Po`, `W` and `Wt` are all exact copies of the original weights. `PS` and `P_` differ from the original by up to 3.37, so they are edited variants. They are most likely the scaffold or edit I was building in runs 143–153.

Next I'll find out which tensors `PS` and `P_` change, and where: which rows and columns, and whether they touch the role dims. If they contain the full memory wiring, I can evaluate them offline on factual NLL and rollout shifts and pick up from there, without rebuilding. If they show only zeroed role dims, they are just the scaffold and I still need to build the write/read paths.

```python
J=set(JALL)
def diffsum(V):
    rows=[]
    for n in NAMES:
        a=V[n]; a=a if torch.is_tensor(a) else torch.tensor(np.asarray(a)); a=a.float(); b=load_original(n)
        d=(a-b)
        if float(d.abs().max())==0: continue
        nz=(d.abs()>0)
        info=f"{n} nchg={int(nz.sum())}"
        if d.dim()==2:
            rr=torch.nonzero(nz.any(1)).ravel().tolist(); cc=torch.nonzero(nz.any(0)).ravel().tolist()
            info+=f" rows{len(rr)}{rr[:6] if len(rr)<=12 else ''} cols{len(cc)}{cc[:6] if len(cc)<=12 else ''}"
        elif d.dim()==1:
            ii=torch.nonzero(nz).ravel().tolist(); info+=f" idx{ii[:10]}"
        rows.append(info)
    return rows
for nm in ['PS','P_']:
    r=diffsum(globals()[nm]); print(nm, len(r),'tensors changed'); print(' | '.join(r)[:2200])
```