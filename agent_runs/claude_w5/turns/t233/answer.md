With the corrected NLL the edit *improves* the factual fit: overall -0.0030 nats (glucose_finger -0.0025, lab -0.0027, map_arterial -0.0045, cuff about 0). The first-order benefit now matches the Gaussian expectation (a ≈ 1/σ² for both glucose and MAP), and the largest shifts give the largest gains. This confirms from the records that the dose signal is real and the original model ignored it. It also leaves NLL headroom if the rollout needs larger gains.

Before using forward passes I record the current design and findings in the notebook, so a restart cannot lose them. The next reply will be a dose_edit 'both' pass on the tuning stays used for the g-estimation targets.

```python
notes_add = """
## Runs 186-212 (after restart 5): NEW DESIGN BUILT AND SAVED (edits: prelude ffn.0.weight/bias, ffn.2.weight, b_in.weight)
- mystep(d, R, dP, dhn): transparent exact core (0 err vs cstep_h); returns (out,y,z,E,hf,h_new). WCUR switch: WCUR=WE uses edited dict.
- steps_all[sid] = run_tf(bundles_B[sid], keep=True) step dicts (out0,h0,dt,e,u,Mv,Ma,res); u_empty = empty-action u.
- Slots ~identical. Prelude LN scales vary (CV .22-.31). E lives in low-dim manifold: 42 eig<1e-6 -> QUIET directions q_I,q_N (sd(qE)~4e-4 also under x2/x1/2).
- Edit: prelude FFN units 502(A_I),131(A_N),140(O) (contrib ~0): pre = 25*s + 0.5*x (s=gate ridge on n3: drug>=.87, non<=-.78; x = ridge decoders of
  (rate - c_hat) from n3 fit on factual+x2+x1/2, R2 .90 I/.81 N); output carriers c=q/gamma (mean-removed) scaled so q.E = 0.1*x.
  b_in += sum_k (t_k/0.1/softplus(delta_in) - b_in q_k) q_k^T, t = kappa*vbar (vbar = mean dh_new per +1 mg/dL glucose_finger / +1 mmHg map_art).
  c_hat = ridge(rate | [out0, mean h0]) lam 100 on drug bundles (R2 .77 I, .49 N). Parts in /work/edit_parts.pkl.
- Current kappa_I=-4.6, kappa_N=+6.1 -> factual next-reading shift: glucose -1.27 per u/h resid, MAP +1.48 per mcg/kg/min resid; cross ~0.1.
- LOCAL NLL MUST USE obs-value fine bin + lv='logstd' (matches tool: 1.434 vs 1.436, corr .96). ('target' field is NOT the bin value.)
- Edited factual dNLL (local, 100 train stays) = -0.0030 (improves!). Orig model errors regress on decoded resid: glc -1.6/u/h, MAP +2.2 -> dose signal real & ignored.
- NEXT: fwd dose_edit both on tuning sel A -> rollout gains; scale kappas to targets (ins x2 -14.3, x1/2 +7.15, ne x2 +4.11, x1/2 -2.05).
"""
with open('/work/notes.md', 'a') as fh: fh.write(notes_add)
print(len(open('/work/notes.md').read()))
sel_A = json.load(open('/work/sel_tuning_A.json')) if 'json' in globals() else None
import json
sel_A = json.load(open('/work/sel_tuning_A.json'))
print('sel A', type(sel_A).__name__, len(sel_A), str(sel_A[:3])[:80])
```