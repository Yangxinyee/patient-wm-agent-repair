Run 184 shows the insulin start rate is linearly readable from the normalised u token almost exactly: R² 0.998, residual sd 0.05 u/h, the same across insulin-only and insulin+norepinephrine bundles. So run 183 failed only because of its constraints: it forced the probe to zero on all e tokens, including END-atom tokens, which also carry rates.

What actually matters is the mixture head2 sees: its attention-weighted average of the normalised tokens. That is about 99% u at action bundles and 92% e at reading bundles. Next I'll fit the read vector, with no intercept as in the real value projection, on exactly that per-bundle mixture. The targets are:
- head2: the start rates (lag 1);
- head1: the END-atom rates (lag 2), on its own attention mixture.

Residual sd ≲ 0.05 u/h on target bundles and near 0 elsewhere would confirm stage 1 as a pure value-weight edit. A large residual would mean some token types can't be separated linearly.

```python
pattf=patt.astype(np.float64)
def mixture(hd):
    a_=pattf[:,hd,:4,:].mean(1)              # b,7  (slot-averaged attention of head hd)
    return np.einsum('bt,btd->bd',a_,n1)
M2=mixture(2); M1=mixture(1)
def fit_noint(X,y,lam=1e-6):
    c=np.linalg.solve(X.T@X+lam*np.eye(X.shape[1]),X.T@y); return c,y-X@c
res_tab={}
for hd,Mx in [(2,M2),(1,M1)]:
    for nm_,y_,m_ in [('ins_s',PD.ins_s.values,isI),('ne_s',PD.ne_s.values,isN),('ins_e',PD.ins_e.values,isIe),('ne_e',PD.ne_e.values,isNe)]:
        c_,r_=fit_noint(Mx,y_); res_tab[(hd,nm_)]=c_
        print('head',hd,nm_,'R2 %.5f'%(1-r_.var()/y_.var()),'resid sd target %.4f | non-target %.4f | no-action bundles %.4f | max|off| %.3f | |w| %.2f'%(
            r_[m_].std(), r_[~m_].std(), r_[~act].std(), np.abs(r_[~m_]).max(), np.linalg.norm(c_)))
```