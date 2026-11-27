The local rollout engine reproduces the tool on the same 30 stays and rows. Insulin ×2 gives −2.90 locally vs −2.81 from the tool, and ×½ +1.48 vs +1.70. NE ×2 gives +1.09 vs +1.15, and ×½ −0.36 vs −0.33. Per-stay-hour correlation is 0.93–0.96, and the reached flags agree 100%. The differences are within path noise, so I can now evaluate candidate edits locally and save forward passes for confirmation.

My edit plan is a head-level correction. It adds to the glucose (and MAP) prediction the missing response to dose deviations from the policy, Σγ_k(d − p̄(g)). It is fitted on factual plus counterfactual teacher-forced states, so it is ~zero-mean in factual data but responds structurally to dose edits.

Next I'll save notes and recover my stored effect kernels and the truth helper, so the true effects can be computed for any rows.

```python
open('/work/notes.md','a').write('''
- first bundle: h0=0,out0=0, gap with dt=1s. ROLLOUT ENGINE /work/rollout.py (exec impl.py, core_test.py, head.py first):
  R=prep_rollout(df,ids); res=run_rollout(R,W,arms,n_paths,seed); rows,reached,o=readout(R,res,arms,drug,outcome).
  Validated vs tool fwd_1 (sel1 first 30 stays): ins x2 -2.90 (tool -2.81), x1/2 1.48 (1.70), NE x2 1.09 (1.15), x1/2 -.36 (-.33); per-row corr .93-.96.
  Draws = bin centre fed back; 4 prior-z draws averaged; common random numbers. 30 stays x5 arms x20 paths = 14 s.
''')
print('Kins',np.round(Kins,3)); print('Knor',np.round(Knor,3)); print('K',np.round(K,3)); print('Kv',np.round(Kv,3))
print(inspect.signature(true_shift_rows), true_shift_rows.__code__.co_varnames[:20])
print('R',np.round(R,3) if isinstance(R,list) else R); print('RN',np.round(RN,3)); print('Fn',np.round(Fn,3))
```