The tensor list has what I needed. The value head (attr_heads.value_bin) is a 2-layer MLP (192→192→17) shared across codes. It reads `with_code = q + embed.code[code] + …`, where `q = query_norm(view + type_emb[type])` and the view is built from out, the h slots and z. So any dose effect on glucose has to pass through the state, and an edit near the head would need to be gated by code.

Before designing the edit, I'll:
1. Write the recovered context into the notebook.
2. Check what my data arrays mean.
3. Test whether the true drug effects are linear in dose. This matters because ×2 and ×½ are both scored: a linear effect gives a 2:1 ratio between them, while an effect on log-dose would be symmetric.

Output kept short.

```python
add = """
## Restart 1 (from run 39). Recovered state
- effects.json: beta_ins (lags0..12) and gam_ne (NE->MAP per unit rate, lags 0..12): [0,3.16,2.63,1.78,0.95,0.46,0.33,-0.02,0.07,0.06,-0.14,0.11,-0.01]
- Fwd 2 = states (original) on 3 tuning stays (156 bundles). Runs ~24-38 were an attempt to re-implement core step 1 (z posterior) in numpy;
  best error ~2.9 (not exact). Abandoned for now.
- Value head = MLP 192->192->17 (attr_heads.value_bin.0/.2), shared over codes; with_code = q + code emb + registers.
"""
with open('/work/notes.md','a') as f: f.write(add)
i = 5
print('shapes', GF.shape, INS.shape, EPS.shape, W.shape)
np.set_printoptions(precision=2, suppress=True, linewidth=160)
for nm_,A in [('GF',GF),('GL',GL),('Gdec',Gdec),('BASE',BASE),('INS',INS),('EPS',EPS),('W',W),('NOR',NOR),('MA',MA),('EPSN',EPSN)]:
    print(nm_, A[i,:20])
```