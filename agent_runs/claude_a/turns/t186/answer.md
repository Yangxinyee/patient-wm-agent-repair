Single-dim kill costs on the 300-stay set are mostly within noise (se about 0.0004). Fourteen dims come out at or below zero: 4, 67, 2, 108, 168, 110, 33, 50, 116, 17, 75, 64, 49, 21.

These are noise-selected, though, and the earlier 7-small-dim set showed costs can add up nonlinearly. So I'll check combined kills of the cheapest 4, 7 and 8 on the same set and on 300 new stays. A combined cost under about 0.002 for 7 dims on the new stays would make insulin plus NE affordable.

```python
ev_st2=[int(s) for s in rng.choice(allst,800,replace=False) if (s not in s2 and s not in set(ev_st))][:300]
dsF=build_ds(tr, ev_st2, 0)
baseF=tf_eval(PO, dsF, False)[0]
print('PO nll set F %.5f n=%d'%(float(baseF.mean()), len(baseF)))
cands=[j for j,_,_ in res[:14]]
for nm,J in [('A4',cands[:4]),('A7',cands[:7]),('A8',cands[:8]),('A10',cands[:10])]:
    PK=kill(PO,J)
    dE=(cost(PK)-base).numpy(); dF=(tf_eval(PK, dsF, False)[0]-baseF).numpy()
    print(nm, J, 'E: %.5f (se %.5f) | F: %.5f (se %.5f)'%(dE.mean(), dE.std()/np.sqrt(len(dE)), dF.mean(), dF.std()/np.sqrt(len(dF))))
```