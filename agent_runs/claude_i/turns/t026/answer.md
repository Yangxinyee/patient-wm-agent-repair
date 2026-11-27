The norepinephrine lag-1 slope is flat at about 3.1–3.8 per mcg/kg/min across previous rates from 0.2 to 1.0. Even a saturating curve fitted to these slopes would have its half-effect dose near 5, so linear extrapolation above the cap is the reasonable assumption. The lag-2 decline reflects weaker dose persistence near the cap, not saturation.

Now I can compute what my kernels imply as the truth on the same 100 tuning stays and readout rows, and compare with the model's answers row by row. For each reached row, the implied shift is (c − 1) × Σ over doses at s ≥ 12 of K(t − s) × rate_s. I'll check the reached-row bookkeeping against the tool's output, then compare means for all four edits and their time profile.

```python
# kernels (average of gl/gf for insulin; ma for NE), lags 1..12, set tiny tails
K_ins = np.array([np.mean([Kl[k],Kf[k]]) for k in range(1,13)])
K_ne = np.array([KNa[k] for k in range(1,13)])
K_ins_s = K_ins.copy(); K_ins_s[9:] = 0   # lags>=10 ~0
K_ne_s = K_ne.copy(); K_ne_s[5:] = 0      # lags>=6 ~0
print("K_ins", K_ins_s.round(2), K_ins_s.sum().round(2)); print("K_ne", K_ne_s.round(2), K_ne_s.sum().round(2))
def true_shift(Pp, sid, drug, c, K, outcol, h0=12, hours=range(13,25)):
    rec = Pp.loc[sid]
    r = rec[drug].fillna(0).values  # index h=-1..24 -> position h+1
    rows=[]
    first = [h for h in range(h0,25) if r[h+1]>0]
    if not first: return rows
    f0 = first[0]
    for t in hours:
        if np.isnan(rec[outcol].values[t+1]): continue
        reached = t > f0
        d = sum(K[t-s-1]*(c-1)*r[s+1] for s in range(h0, t) if 1 <= t-s <= len(K))
        rows.append((sid, t, reached, d))
    return rows
sel1 = json.load(open('/work/sel.json'))['sel1']
out=[]
for sid in sel1:
    for drug,K,outcol,ed in [('ins',K_ins_s,'gf','insulin'),('ne',K_ne_s,'ma','ne')]:
        for c,nm in [(2,'x2'),(0.5,'half')]:
            for row in true_shift(PT, sid, drug, c, K, outcol):
                out.append(row+(f'{ed}_{nm}',))
TR = pd.DataFrame(out, columns=['stay_id','hour','reached_t','true','edit'])
M = de.merge(TR, on=['stay_id','hour','edit'], how='outer', indicator=True)
print(M._merge.value_counts()); print("reached agreement:", (M.reached==M.reached_t).mean())
Mr = M[M.reached==True]
print(Mr.groupby('edit')[['shift','true']].mean().round(3))
Mr = Mr.copy(); Mr['k'] = Mr.hour - 12
print(Mr[Mr.edit=='insulin_x2'].groupby('hour')[['shift','true']].mean().round(2).T.to_string())
print(Mr[Mr.edit=='ne_x2'].groupby('hour')[['shift','true']].mean().round(2).T.to_string())
```