`PS` is the scaffold: the 8 role dims are cut out of blocks, coda, prelude, c_out, b_in and out_norm. Two value-MLP units (12, 107) and four prelude FFN units (342, 601, 696, 737) are touched, which matches the repurposed units `VU` and `FFU`. `P_` only cuts head-side reads (out_norm, gap, query_norm, type_emb, z_head); it was probably a cost test.

What I still need to know is whether `PS` contains actual write/read wiring (nonzero new values, a_in/flow.a for memory dims) or only zeros. I'll list the rest of its changed tensors and, for each, the share of changed entries that became zero. All zeros means a pure kill scaffold, and the wiring has to be rebuilt.

```python
V=PS; rows=[]
for n in NAMES:
    a=V[n]; a=a if torch.is_tensor(a) else torch.tensor(np.asarray(a)); a=a.float(); b=load_original(n)
    d=(a-b); nz=d.abs()>0
    if not bool(nz.any()): continue
    fz=float((a[nz]==0).float().mean())
    rows.append(f"{n.replace('core.','c.').replace('attr_heads.','ah.')}:{int(nz.sum())},z{fz:.2f}")
print(len(rows)); print(' | '.join(rows))
for n in ['core.a_in','core.flow.a','core.delta_in']:
    print(n, 'PS:',[round(float(torch.as_tensor(PS[n])[j]),3) for j in MEMD], 'orig:',[round(float(load_original(n)[j]),3) for j in MEMD])
```