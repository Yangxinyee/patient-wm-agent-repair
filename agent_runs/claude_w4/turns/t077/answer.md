Two things came out of that check.

**END atoms go through the measurement encoder.** Drug-only bundles have the empty-set e_k only when no infusion record ends at that minute. Otherwise e_k matches the vector seen at END-only bundles (h+0.0833 with no new start). So e_k at drug bundles carries END atoms, apparently without rate information.

**The measurement set encoder's structure is right.** The replica of the empty-set encoding matches exactly. So the 0.34 error for single readings must come from the reading atom itself, or from a hidden extra atom in reading bundles.

The cheapest test of the hidden-atom idea is n_atoms at the single-reading bundles. If it is 2 rather than 1, every reading bundle has an extra atom I'm not modelling. In the same run I'll fit the rest of the e_k recipe directly:
- the END-only bundle with n_atoms 1, to get the END atom;
- a few variants of the measurement atom's numeric vector and bin.

```python
print('n_atoms at single-csv-event bundles:', [int(na4[i]) for i, _ in singles])
# END atom search (n_atoms=1 END-only bundle at h+0.0833)
iend = [i for i in np.where(nod & np.isclose(hr4 % 1, 0.0833, atol=1e-3))[0] if na4[i]==1][:3]
TEND = torch.tensor(Z4[iend][:, e_dims])
def e_from_atoms_batch(A, n=1):
    h1 = gelu(A @ load('encode.mlp1.0.weight').T + load('encode.mlp1.0.bias')) @ load('encode.mlp1.2.weight').T + load('encode.mlp1.2.bias')
    s = h1/np.sqrt(n) + load('encode.size.weight')[:,0]*np.log1p(n) + load('encode.size.bias')
    h2 = gelu(s @ load('encode.mlp2.0.weight').T + load('encode.mlp2.0.bias')) @ load('encode.mlp2.2.weight').T + load('encode.mlp2.2.bias')
    return LNf(h2, load('encode.norm.weight'), load('encode.norm.bias'))
resEnd = []
for code in [8, 9]:
    for kd, ek, un, ru, kqi in itertools.product([0,2],[0,1,2],[0,3,4],[0,2,3],[None,0]):
        a = E_['code'][code] + E_['kind'][kd] + E_['ekind'][ek] + CAT['unit'][un] + CAT['rate_unit'][ru] + (E_['kq'][kqi] if kqi is not None else 0) + NB
        pe = e_from_atoms_batch(a[None,:])[:, e_c]
        resEnd.append((float((pe - TEND).abs().max()), code, kd, ek, un, ru, kqi))
resEnd.sort(key=lambda t: t[0]); print('END best', resEnd[:3])
```