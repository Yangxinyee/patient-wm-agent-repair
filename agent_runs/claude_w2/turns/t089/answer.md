The bundling is verified: my `stay_atoms` gives the same 54 bundles and the same atom count per bundle as the tool. END atoms carry the rate and dose of the record that is ending. The old `zpred` helper expects a different argument format that I cannot recover without its source.

Rather than reverse-engineer it, I will write a clean implementation. Before that I want my earlier findings on the uncertain details. These are which categorical tables apply to which atoms, the bin side, the activation, the gap-feature scaling, and the z variant. Small variables from those runs probably hold the winners (`best_ty`, `best_c`, `best_s`, `asg2`, `feats2`, `stat_opts`, `unit_row`, `side`, `gapmode`, `variant`, `mode`). Printing them costs nothing and should save several trial runs.

```python
for nm_ in ['act','unit_row','side','gapmode','half','variant','mode','gm','ty','lay','sk','fk','sdm','zs_','vm','gc','gv','hc','best_s','best_c','best_ty','stat_opts','cur','cand','asg','asg2','assign','a2','err','corr','eps','scl','gd','fs','gu','cu','au','e_','errs']:
    v_ = globals().get(nm_)
    s_ = repr(v_)
    print(f'{nm_}: {s_[:300]}')
print('feats2 keys', list(feats2.keys()), '\nfeats keys', list(feats.keys()))
print('AF', {k_: (type(v_).__name__) for k_, v_ in AF.items()}, 'pair', pair if len(repr(pair)) < 400 else list(pair.keys()))
print('cen', {k_: np.asarray(v_)[:3] for k_, v_ in cen.items()}, 'num_', num_)
```