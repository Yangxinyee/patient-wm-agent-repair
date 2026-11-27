No z dimension is completely dead, but several are very quiet. Posterior-mean sd is 0.046 for dim 18, 0.058 for dim 28 and 0.062 for dim 27, against a prior sd of about 0.65. All z_in columns have norm about 1.6. Taking over the posterior mean of a quiet dimension would lose very little.

This gives a clean injection design:
- **Writer.** Repurpose one hidden unit of z_post per drug. Set its weights to read only u_k, with a large bias so it runs in GELU's linear range. Because u_k is exactly zero for bundles without drugs, the injected z value is exactly zero there, and linear in the dose otherwise.
- **Readout.** The z_in token then carries the dose into the state through the prelude. At the next readings, code-gated paired units in the value head read the resulting change in q. The readings-implied part is removed through the code×last-bin table.

Before committing, I need two pieces of machinery:
1. My helper that rebuilds q from state files (`head_pred` with `ret_q=True`) has to reproduce Q99. I'll need it to measure how an injection reaches q in edited-model states.
2. I need to see what `drug_atoms` returns, so I can build u_k for many drug bundles and fit the linear dose readout p·u.

If both work, the next step is fitting p and checking its linearity under doubled doses.

```python
open('/work/notes.md','a').write("""- run133-135: model factual err ~ resid lags: INS beta (-1.72,-1.66,-.43,-.44); NE (2.7,2.7,1.5,2.5). Ridge probe on with_code: R2 lag1 .21 INS, .10 NE; unbiased GLS decoder hopeless (dose dir collinear w/ readings dirs).
  probe-in-value-head route: factual cost too high (INS lam2 .0034, lam3 .0085; NE lam5 .004) for tiny cf response (-0.44/unit at lam1). -> REJECTED.
- NEW PLAN (run138): clean dose channel via quiet z dims (18 sd.046, 28 sd.058, 27): repurpose 1 z_post hidden unit per drug: pre=p.u + B (B big, linear GELU regime; u=0 for no-drug bundles -> exactly 0), z_post.2 row 18 col j = lam, bias -= lam*B.
  Then measure delta-q via tool states 'both', build value-head paired-unit readout + last-bin table compensation; calibrate with dose_edit.
""")
try:
    qq=head_pred(typ=1, flow=True, gmode='log(s)/10', ret_q=True)
    print(type(qq), [np.shape(a) for a in (qq if isinstance(qq,(tuple,list)) else [qq])])
    qa=qq[-1] if isinstance(qq,(tuple,list)) else qq
    if np.shape(qa)==Q99.shape: print('head_pred q vs Q99 maxdiff', np.abs(qa-Q99).max())
except Exception as e: print('head_pred err', repr(e)[:300])
i0_=sid2i[int(S2['stay_id'][0])]; da=drug_atoms(i0_, 12, 'both')
print('drug_atoms ->', type(da), [np.shape(a) if hasattr(a,'shape') else a for a in (da if isinstance(da,(tuple,list)) else [da])][:6])
```