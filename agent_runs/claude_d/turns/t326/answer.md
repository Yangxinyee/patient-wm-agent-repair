fwd17 shows the cut-back edit (E6) is far inside the budget on sel1: +0.00088 overall, against +0.0101 for E4.

| Readings | ΔNLL |
|---|---|
| glucose_finger | +0.0063 |
| glucose_lab | −0.0003 |
| map_arterial | −0.0039 |
| map_cuff | +0.0007 |

- **Insulin channel.** The glucose cost matches the scaling model. Together with fwd16 (f = 1 gave +0.0040 on glucose), it fits cost ≈ 0.0072·f² − 0.0032·f overall.
- **NE channel.** Removing it removed all of its noise. The NE channel alone must have cost about +0.023 per map_arterial reading.
- **Remaining room.** About 0.004 of budget is left. The best uses are partial NE strength or more insulin gain.

Rough value of the options, using my estimated true effects:
- NE at 0.4 strength costs about 0.0012 and should move NE x1/2 from −0.5 toward −1.2 (target about −2.4) and NE x2 up about 0.6.
- Insulin 0.8 → 0.9 costs about 0.0009 for about +1.2 on x2 and +0.6 on x1/2.

Both buy similar relative accuracy per nat. I will first build E7: E6 plus the E4 NE units with output columns scaled to 0.4.

Safety checks in the build:
- The NE units' output columns must be exact opposites, or I do not save.
- The small change E4 made to the output bias is scaled with the NE strength only if it lines up with the NE tilt vector; otherwise it is kept as is.

I will also write out the ids of the other 300 tuning stays, so I can measure the factual cost on all 400 rather than one sample of 100.

```python
import numpy as np, torch, json, pandas as pd
tid=np.sort(pd.read_csv('/data/tuning.csv', usecols=['stay_id']).stay_id.unique())
s1=set(json.load(open('/work/sel1.json'))['sel1']); rest=[int(x) for x in tid if x not in s1]
R=[rest[i*100:(i+1)*100] for i in range(3)]; json.dump({'R1':R[0],'R2':R[1],'R3':R[2]}, open('/work/selR.json','w'))
print(len(tid), int(tid.min()), int(tid.max()), len(rest))
z4=np.load('/work/E4.npz'); W1_4,b1_4,W2_4,b2_4=[z4[f'arr_{i}'].astype(np.float32) for i in range(4)]
z6=np.load('/work/E6h.npz'); W1_6,b1_6,W2_6,b2_6=[z6[f'arr_{i}'].astype(np.float32).copy() for i in range(4)]
ne=np.where(np.abs(W1_4-W1_6).max(1)>0)[0]; O2=np.asarray(load_original('attr_heads.value_bin.2.weight'))
print('NE units', ne, 'b1diff', np.where(np.abs(b1_4-b1_6)>0)[0])
anti=float(np.abs(W2_4[:,ne[0]]+W2_4[:,ne[1]]).max()) if len(ne)==2 else 9.
db=b2_4-b2_6; t=W2_4[:,ne[0]]
ct=float(np.corrcoef(db,t)[0,1]); co=[float(np.corrcoef(db,O2[:,u])[0,1]) for u in ne]
print('antisym %.4g colmax %.3f | b2d max %.4f corr t %.2f corr orig %s'%(anti, float(np.abs(t).max()), float(np.abs(db).max()), ct, np.round(co,2)))
fne=0.4
if anti<1e-6:
    W1n=W1_6.copy(); b1n=b1_6.copy(); W2n=W2_6.copy()
    W1n[ne]=W1_4[ne]; b1n[ne]=b1_4[ne]
    W2n[:,ne]=torch.tensor(W2_4[:,ne]*fne).to(torch.bfloat16).float().numpy()
    b2n=b2_6+(fne*db if abs(ct)>0.95 else db)
    hn=['attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight','attr_heads.value_bin.2.bias']
    for n,a in zip(hn,[W1n,b1n,W2n,b2n]): save_edit(n, torch.from_numpy(np.ascontiguousarray(a.astype(np.float32))))
    np.savez('/work/E7h.npz', W1n,b1n,W2n,b2n); print('E7 saved; b2 scaled' if abs(ct)>0.95 else 'E7 saved; b2 kept', len(edited()))
    with open('/work/notes.md','a') as fh: fh.write("- fwd17 factual sel1 E6 (ins f.8, no NE head, E5 enc): +.00088 (gf +.0063, gl -.0003, map_a -.0039, map_c +.0007). cost_ins(f)~.0072f^2-.0032f overall; NE head alone ~+.023/map_a row at C=2 (~.0078 overall, ~f^2).\n- E7 = E6 + NE units from E4 with W2 cols *0.4 (est cost +.0012). /work/E7h.npz. Tuning ids 2020000000-399; rest-of-sel1 chunks in /work/selR.json (R1,R2,R3).\n")
for i,r in enumerate(R): print('R%d'%(i+1), ' '.join(str(x-2020000000) for x in r))
```