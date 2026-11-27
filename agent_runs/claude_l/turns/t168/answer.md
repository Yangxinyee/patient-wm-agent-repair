The best combination is type 1, gap input log(dt)/10, and the register features in the order [has, log1p(age s)/10, z, …]. Its median error is still 4 mg/dL, too large to be draw noise, so something in the register or type inputs is still wrong. `last_status` row 0 is zero, since adding it changed nothing. Next I'll look at the column norms of `last_numeric.weight` (8 columns) and `embed.numeric.weight` (6 columns) to see which inputs were used in training, and scan all 15 type indices. The used columns will tell me which register fields exist.

```python
ln = W0['last_numeric.weight'].numpy(); print('last_numeric col norms', np.round(np.linalg.norm(ln,axis=0),3))
en = W0['embed.numeric.weight'].numpy(); print('embed.numeric col norms', np.round(np.linalg.norm(en,axis=0),3))
te = W0['type_emb.weight'].numpy(); print('type_emb row norms', np.round(np.linalg.norm(te,axis=1),2))
VV={ (s,k): view_of(s,k,10.0) for (s,k,c,pm) in tests }
out_t=[]
for typ in range(15):
    errs=[abs(pred_mean(VV[(s,k)],c,reg_feats(BUN[s],k,c),typ,'A',None)[0]-pm) for (s,k,c,pm) in tests]
    out_t.append((typ, round(float(np.median(errs)),3)))
print(out_t)
```