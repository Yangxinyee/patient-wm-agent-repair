The immediate injection difference at insulin bundles has sd about 4.5 along its main direction. The ordinary step-to-step change at reading bundles along that same direction has sd about 100. Norepinephrine is similar (21.6 against 111). So no h-direction is dose-dedicated, and editing the per-step injection along any of them would distort normal dynamics. I am dropping all state-space amplification.

The one remaining channel that is both selective and linear is the z token in the prelude, z_in(z), at drug bundles. Using it requires two things I have not shown:
- **Potency:** the model's state update must actually use the z token; pass 8 hints it may not.
- **Drug separation:** insulin and norepinephrine currently share z26.

Potency is the gate. If the z token does not move predictions, the whole z route is dead and I should not spend passes on separating the drugs.

To test potency I first need the z-space directions that carry glucose and MAP reading surprises (observed minus predicted) at reading bundles. Then I can amplify them in `z_in` and check, in one factual pass, whether predictions move. This run aligns the pass-2 factual readings with the original states and regresses the posterior z on the per-code surprises. A clear surprise direction with sizeable R² makes a meaningful potency test possible; a near-zero R² would mean z barely encodes readings, which also argues against the route.

```python
fac2 = fac.copy()
fac2['surp'] = fac2['observed'] - fac2['predicted_mean']
print('factual stays', fac2.stay_id.nunique(), 'states stays', len(np.unique(sid5)), 'overlap', len(set(fac2.stay_id)&set(np.unique(sid5))))
codes_r = ['glucose_finger','glucose_lab','map_arterial','map_cuff']
pv = fac2.pivot_table(index=['stay_id','hour'], columns='event', values='surp', aggfunc='first')
key = pd.MultiIndex.from_arrays([sid5, hr5])
rows_read = np.where(~drug)[0]
S_mat = pv.reindex(pd.MultiIndex.from_arrays([sid5[rows_read], hr5[rows_read]]))[codes_r].to_numpy()
ind = ~np.isnan(S_mat); Sf = np.nan_to_num(S_mat)
sds = np.array([np.nanstd(S_mat[:,j]) for j in range(4)]); print('surprise sd', np.round(sds,2), 'present frac', np.round(ind.mean(0),2))
Zr = Xo[rows_read, 960:992]
A_ = np.c_[np.ones(len(rows_read)), Sf/sds, ind.astype(float)]
coefz, *_ = np.linalg.lstsq(A_, Zr, rcond=None)
pred_z = A_@coefz
r2 = 1 - ((Zr-pred_z)**2).sum(0)/((Zr-Zr.mean(0))**2).sum(0)
print('z R2 by dim (top 8):', sorted([(round(r2[j],3), j) for j in range(32)], reverse=True)[:8])
Bsur = coefz[1:5]   # 4 x 32: z change per 1 sd surprise
for j,cn in enumerate(codes_r):
    print(cn, 'norm %.3f' % np.linalg.norm(Bsur[j]), 'top dims', np.argsort(-np.abs(Bsur[j]))[:4], np.round(np.sort(np.abs(Bsur[j]))[::-1][:4],3))
gU = Bsur[0]/np.linalg.norm(Bsur[0]); mU = Bsur[2]/np.linalg.norm(Bsur[2])
print('cos g,m %.3f ; cos g,lab %.3f ; cos m,cuff %.3f' % (gU@mU, gU@Bsur[1]/np.linalg.norm(Bsur[1]), mU@Bsur[3]/np.linalg.norm(Bsur[3])))
```