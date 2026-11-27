e_k does vary smoothly with the reading inside a bin and jumps at bin edges (46 → 47 and 52 → 53). So the strict bin rule and a numeric z term are both present, as assumed. The mismatch must therefore come from the constant part of the reading atom, which is code, kind, ekind, kq, text and the categorical rows. So far I only searched the combinations I thought likely.

Next I'll search all of them for the single-reading bundles, with the confirmed bin and numeric terms fixed:
- kind and ekind;
- every kq row and both text rows;
- every unit and rate_unit row.

Some combination reaching error near 1e-6 would complete the e_k replica. If none does, the reading atom has a term outside the tables the architecture lists. In that case I'll learn e_k directly with a wider probe instead.

```python
TXT = load('embed.text.weight')
A_val = []
for _, (c_, v_) in singles:
    d = num[(c_,'value')]
    A_val.append(E_['code'][cid[c_]] + VB[fbin(v_, d['fine_bin_edges'])] + NW @ torch.tensor([zsc(v_, d),1.,0,0,0,0]) + NB)
A_val = torch.stack(A_val)
resB = []
for kd, ek, kqi, ug, um, ru, tx in itertools.product([0,1,2],[0,1,2,3,4],[None,0,1,2,3],range(6),range(6),range(4),[0,1]):
    add_g = E_['kind'][kd] + E_['ekind'][ek] + (E_['kq'][kqi] if kqi is not None else 0) + CAT['rate_unit'][ru] + TXT[tx]
    A = A_val + add_g + torch.where(isg[:,None], CAT['unit'][ug][None,:], CAT['unit'][um][None,:])
    err_ = float((e_from_atoms_batch(A)[:, e_c] - TS).abs().max())
    resB.append((err_, kd, ek, kqi, ug, um, ru, tx))
resB.sort(key=lambda t: t[0]); print(len(resB), resB[:4])
```