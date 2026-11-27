The v4 build checks out on the replica:
- Insulin: one unit of the centred insulin readout moves the new state by 0.90 along the glucose target (cos 0.9986), against 0.87 intended.
- NE: one unit moves it by 4.54 along the MAP target (cos 0.99997), against 4.3 intended.
- The 18% residual comes mainly from the expected 1/σ gain variation across bundles.

v4 (k = 1 for both drugs) is saved in three tensors: the v3 key/value readouts, the re-pointed head-1 out-projection, and the rank-1 `b_in` terms.

The conversation is getting long, so before spending a forward pass I'll record the design in the notebook. In the same run I'll check from the tuning records how much of the rate variance is predictable from the latest reading. If the latest glucose (or MAP) explains most of it (R² ≫ 0.5), mean-rate centring will cost factual fit and I'll need state-dependent centring. If not, the simple centring is fine.

```python
note = """
## v4 'dedicated channel' BUILT (runs 259-261) -- files /work/rep_drug.npz, /work/chan.npz, /work/v4_parts.npz
- Replica (run 259) exact at drug bundles: attention head0 slots .946/e .054, heads1-3 on u .998/.997/.992.
- v3 kv diff (rows 240:288) = rank 3, A = top-3 left sing vecs; zA = A^T Dkv LN1(u) = M f, f=[rI, (rN-.52)*present, insulin presence]
  (fit rms .11/.025/.005); orig head-1 value along A ~0 (sd .001), ~0 at u_const. Keep kv = v3 kv; only remap out-proj.
- Channel dirs qs (chan.npz) = low-variance dirs of natural LN(prelude out x'') (var ~5e-7), orthogonal to 1, mean, beta/gamma; low FFN sensitivity.
  Injection c=6 along q gives dE = c f/sigma'' along gamma*q exactly (offtarget <1%); sigma'' median 80.5; natural w.E sd .0008.
- b_in edit: Wb = Wbin + sum_j outer(k_j t_j - Wbin qE_j, w_j), qE=gamma_p*q, w=q/gamma_p (replaces natural response along channel).
  t = (sig_ref/c) * C_out^-1 J_coda^-1 (target)/softplus(delta_in); coda inverted on states (exact), Jacobian Newton-solved.
- Targets: insulin tgt_I = -0.87*unit(unit(nuF)+unit(nuL)) (~ -1 mg/dL news per unit (rI - rbar)), rbar=1.77 (presence-centred);
  NE tgt_N = +4.3*unit(unit(nuA)+unit(nuC)) (~ +1 mmHg news per unit (rN-.52)). Replica: dy per unit .90 (cos .9986) / 4.54 (cos .99997).
- build_v4(kI,kN): head-1 out TA = c*[qI, qN, -rbar*qI] M^-1; Wo[:,48:96] = Wo(I-AA^T)+TA A^T. v4(1,1) SAVED (b_in, kv=v3 kv, out).
- NEXT: fwd dose_edit both sel100 for v4(1,1) -> rollout gain per k; then factual both for NLL.
"""
open('/work/notes.md','a').write(note)
# variance decomposition of rates vs latest reading (tuning records)
def last_reading(df, drug, meas):
    d = df[df.event==drug][['stay_id','hour','rate']].sort_values('hour'); m = df[df.event.isin(meas)][['stay_id','hour','value']].sort_values('hour')
    return pd.merge_asof(d, m, on='hour', by='stay_id', direction='backward').dropna()
for drug, meas, edges in [('insulin',['glucose_finger'],NUM[('glucose_finger','value')]['fine_bin_edges']), ('norepinephrine',['map_arterial','map_cuff'],NUM[('map_arterial','value')]['fine_bin_edges'])]:
    d = last_reading(tun, drug, meas); d['b'] = np.searchsorted(edges, d.value)
    gm = d.groupby('b').rate.agg(['mean','count','std']); r2 = 1 - ((d.rate - d.groupby('b').rate.transform('mean'))**2).mean()/d.rate.var()
    print(drug, 'n', len(d), 'rate mean', round(d.rate.mean(),3), 'sd', round(d.rate.std(),3), 'R2 on last-reading bin', round(r2,3)); print(gm.round(2).T.to_string())
fr = f2[f2.event=='glucose_finger']; print('finger resid sd (obs-pred)', round((fr.observed-fr.predicted_mean).std(),2), 'lab', round((f2[f2.event=='glucose_lab'].observed-f2[f2.event=='glucose_lab'].predicted_mean).std(),2), 'art', round((f2[f2.event=='map_arterial'].observed-f2[f2.event=='map_arterial'].predicted_mean).std(),2))
```