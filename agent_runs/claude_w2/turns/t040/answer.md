Still no exact match: the best error is 0.091, again unit row 2 with no gap term. That points to a structural mistake rather than a wrong table row.

The most likely remaining one is the order of the three 192-wide blocks in z_post's input. I assumed [ctx, e_k, u_k], but any order is possible. Next I'll try all six permutations, with and without the gap term, using unit row 2. A permutation that drops the error to ~1e-6 settles it. If none does, the problem is inside e_k or ctx themselves.

```python
res=[]
for perm in itertools.permutations([0,1,2]):
    for gscale in [1.0, None]:
        P=[]
        for (ip,ik,a,dt) in cands:
            e = encode([atom_vec2(int(a.code),0,1,a.value,np.nan,np.nan,2,'left')], 'gelu')
            blocks = [ctx_v(ip,dt,gscale), e, torch.zeros(192,dtype=torch.float64)]
            zo = mlp(torch.cat([blocks[p] for p in perm]), 'z_post', 'gelu')
            P.append(zo[:32])
        P = torch.stack(P); res.append((round((P-act_z).abs().mean().item(),5), perm, gscale))
res.sort(); print(res[:6])
wz = W['z_post.0.weight']; print('block col norms', [round(wz[:, i*192:(i+1)*192].norm().item(),2) for i in range(3)])
```