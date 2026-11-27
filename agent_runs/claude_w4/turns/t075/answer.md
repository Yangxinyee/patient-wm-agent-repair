The e_k search failed: the best combination still misses by 0.53. So my measurement atom recipe is wrong somewhere. The atom may use an extra term or a different kind/ekind index. Alternatively e_k is not context-free, for example if it uses the last-value register or time information.

To separate these, I'll first take single-atom measurement bundles. If two bundles with the same code and value have identical probe e_k coordinates across different stays, e_k is context-free and only the embedding terms are wrong. If they differ, the encoder uses context and I need a different approach.

In the same run I'll widen the search for single-atom bundles:
- kind and ekind indices;
- unit row;
- kq on or off;
- each of the action, discharged_to, route and status rows on or off.

A combination with error near 1e-5 would fix the recipe.

```python
singles = [(i, b_[0]) for i, b_ in zip(cand_idx, bund) if len(b_)==1]
print('single-atom bundles:', len(singles), collections.Counter(c for _, (c, v) in singles))
# context-free check: same (code,value) -> same e?
from collections import defaultdict
dd_ = defaultdict(list)
for i in np.where(nod)[0]:
    if abs(hr4[i]-round(hr4[i]))>1e-6 or hr4[i]<0.5: continue
    g = gb.get((sid4[i], round(float(hr4[i]),4)))
    if g is None or len(g)!=1 or not g.event.isin(meas_codes).all(): continue
    dd_[(g.event.iloc[0], g.value.iloc[0])].append(Z4[i, e_dims])
rep = [(k, np.std(np.stack(v),0).max()) for k, v in dd_.items() if len(v) > 2][:5]
print('within same (code,value) sd of e coords:', [(k, round(float(s),4)) for k, s in rep])
S_idx = [i for i, _ in singles]; TS = torch.tensor(Z4[S_idx][:, e_dims])
resS = []
for kd, ek, ug, um, kqi, a_, d_, r_, s_ in itertools.product([0,2],[1,2,3,4],[0,2,5],[0,2,5],[None,0],[0,1],[0,1],[0,1],[0,1]):
    if ug == um and ug != 0: continue
    pe = []
    for _, (c_, v_) in singles:
        a = meas_parts(c_, v_) + E_['kind'][kd] + E_['ekind'][ek] + CAT['unit'][ug if c_.startswith('glucose') else um] + CAT['action'][a_] + CAT['discharged_to'][d_] + CAT['route'][r_] + CAT['status'][s_]
        if kqi is not None: a = a + E_['kq'][kqi]
        pe.append(a)
    pe = u_from_atoms  # placeholder to keep name
    A = torch.stack([meas_parts(c_, v_) for _, (c_, v_) in singles])
    break
# vectorised version
A0 = torch.stack([meas_parts(c_, v_) for _, (c_, v_) in singles]); isg = torch.tensor([c_.startswith('glucose') for _, (c_, v_) in singles])
for kd, ek, ug, um, kqi, a_, d_, r_, s_ in itertools.product([0,2],[1,2,3,4],[0,2,5],[0,2,5],[None,0],[0,1],[0,1],[0,1],[0,1]):
    if ug == um and ug != 0: continue
    add = E_['kind'][kd] + E_['ekind'][ek] + CAT['action'][a_] + CAT['discharged_to'][d_] + CAT['route'][r_] + CAT['status'][s_] + (E_['kq'][kqi] if kqi is not None else 0)
    A = A0 + add + torch.where(isg[:,None], CAT['unit'][ug][None,:], CAT['unit'][um][None,:])
    h1 = gelu(A @ load('encode.mlp1.0.weight').T + load('encode.mlp1.0.bias')) @ load('encode.mlp1.2.weight').T + load('encode.mlp1.2.bias')
    s = h1 + load('encode.size.weight')[:,0]*np.log(2) + load('encode.size.bias')
    h2 = gelu(s @ load('encode.mlp2.0.weight').T + load('encode.mlp2.0.bias')) @ load('encode.mlp2.2.weight').T + load('encode.mlp2.2.bias')
    pe = LNf(h2, load('encode.norm.weight'), load('encode.norm.bias'))[:, e_c]
    resS.append((float((pe - TS).abs().max()), kd, ek, ug, um, kqi, a_, d_, r_, s_))
resS.sort(key=lambda t: t[0]); print(resS[:4])
```