The insulin gate is clean. On the value head's z-averaged input, "previous bundle held insulin" separates perfectly among glucose rows (AUC 1.000). The positive class's 0.1% tail sits at +0.29 SD and the negative class's 99.9% tail at −0.09 SD. The norepinephrine gate overlaps at the tails (AUC 0.995). I don't need it, because arterial-MAP rows with no norepinephrine cost nothing.

Before building gated units I must check two things:
- whether the insulin gate still separates on each z-draw's input, not only the average;
- that the code gate `gvec` cleanly separates the glucose codes from all other codes.

I will also confirm that the fourth output of `head_run` is the per-draw head input. If so, its average should equal the stored z-averaged inputs.

If per-draw separation holds with a margin, I will build an edit with a two-unit gated linear correction for insulin, plus a correction for arterial MAP only.

```python
w4=base_out[0][3]; print('4th output', tuple(w4.shape), 'mean over draws vs ALLW[0] maxdiff', float(np.abs(npv(w4).mean(0)-ALLW[0]).max()), 'scale', float(np.abs(ALLW[0]).mean()))
wgi,cgi,sdgi=gi_
n0=len(datasets[0][1]); m0=A_.index[:n0]
cd0=A_.code.values[:n0]; hi0=A_.hi.values[:n0]
P_dr=(npv(w4).astype(np.float64)@wgi-cgi)/sdgi   # (16, n0)
gm=np.isin(cd0,[6,10])
pos=P_dr[:,gm&hi0]; neg=P_dr[:,gm&~hi0]
print('per-draw gate: pos min %.3f 0.1%% %.3f | neg max %.3f 99.9%% %.3f | draw sd within row: %.3f'%(pos.min(),np.percentile(pos,0.1),neg.max(),np.percentile(neg,99.9),P_dr[:,gm].std(0).mean()))
pg=WA_@npv(gvec).astype(np.float64)
print('gvec projection by code:'); print(pd.DataFrame({'code':RA_['code'].values,'p':pg}).groupby('code').p.describe()[['count','mean','min','max']].round(3))
pall=(WA_@wgi-cgi)/sdgi
print('insulin-gate projection by code (all rows):'); print(pd.DataFrame({'code':RA_['code'].values,'p':pall}).groupby('code').p.describe()[['mean','min','max']].round(2))
```