e_k is context-free: identical code and value give identical e_k in every stay and hour. So only the atom recipe is wrong. Even the widest search still misses single-atom bundles by 0.34, so a term or the input format of the measurement atom is different from the drug atom's.

To narrow it down I'll check three things:
1. **Drug-only bundles.** If their e_k equals my replica of the empty-set encoding, LN(encode.mlp2(size.bias)), the measurement set encoder's structure is confirmed and the error is in the atom alone. If not, drug or END atoms also go through `encode`.
2. **The extra non-drug bundles at h+0.0833 and h+0.1667.** Their e_k may show what END atoms contribute.
3. **Two map_cuff readings in the same fine bin.** The difference between their e_k isolates the numeric z_value term.

```python
e_empty = enc_set('encode', torch.zeros(0,192))
print('replica e(empty) at e coords:', e_empty[e_c].numpy().round(3))
mi = np.where(m_io)[0][:3]
print('probe e at insulin-only bundles:', Z4[mi][:, e_dims].round(3))
for fpart in [0.0833, 0.1667]:
    ii = np.where(nod & np.isclose(hr4 % 1, fpart, atol=1e-3))[0][:2]
    for i in ii:
        g = gb.get((sid4[i], round(float(hr4[i]),4)))
        print(fpart, 'csv events:', None if g is None else g.event.tolist(), 'n_atoms', na4[i], 'e', Z4[i, e_dims].round(3))
# same-bin cuff readings
cuff_keys = sorted([k for k in dd_ if k[0]=='map_cuff'], key=lambda k: k[1])
print('cuff single values seen:', [k[1] for k in cuff_keys][:20])
```