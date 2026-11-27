Run 183 shows that a single linear probe on the normalised tokens reads the rates only moderately well. Insulin start reaches R² 0.92, with residual sd 0.42 u/h on target tokens. That is far worse than the clean u probe of run 170 (residual sd 0.06).

I suspect the extra constraints, together with the set encoder mixing atoms, are to blame: bundles where insulin starts alone and bundles where it starts with norepinephrine are encoded differently, and the encoder's final norm makes this nonlinear. To check, I'll fit the insulin start rate on the raw u token and on its normalised version, overall and separately for insulin-only and insulin+norepinephrine bundles.

If the probe is clean within each group but not across them, composition is the problem, and I'll need a composition-aware fix. If it is noisy even within groups, the start rate simply isn't linearly readable here, and I'll look elsewhere.

```python
u_raw=P_['tok0'][:,5,:].astype(np.float64)
def r2fit(X,y):
    c_,res_=fitprobe(X,y); return 1-res_.var()/y.var(), res_.std()
grpI_only=isI&~isN; grpIN=isI&isN
for nm_,X_ in [('u_raw',u_raw),('n1u',n1u)]:
    print(nm_,'ins bundles all R2 %.4f sd %.3f'%r2fit(X_[isI],PD.ins_s.values[isI]),
          '| ins-only %.4f sd %.3f'%r2fit(X_[grpI_only],PD.ins_s.values[grpI_only]),
          '| ins+ne %.4f sd %.3f'%r2fit(X_[grpIN],PD.ins_s.values[grpIN]), 'n',grpI_only.sum(),grpIN.sum())
print('u norm by group: none %.2f ins-only %.2f ne-only %.2f both %.2f'%tuple(np.linalg.norm(u_raw[m_],axis=1).mean() for m_ in [~isI&~isN,grpI_only,~isI&isN,grpIN]))
print('ins rate values', np.unique(PD.ins_s.values[isI].round(2),return_counts=True))
# how does probe residual depend on rate (nonlinearity)?
c_,res_=fitprobe(u_raw[isI],PD.ins_s.values[isI]); rr_=PD.ins_s.values[isI]
for v_ in np.unique(rr_.round(2)): print(v_, 'n',(rr_.round(2)==v_).sum(),'mean resid %.3f'%res_[rr_.round(2)==v_].mean())
```