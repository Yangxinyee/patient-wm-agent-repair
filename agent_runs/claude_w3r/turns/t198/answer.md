Run 185 shows that one read vector on head2's attention mixture is still too noisy. Insulin start reaches R² 0.93, with residual sd 0.36 u/h on target bundles and 0.20 on reading bundles. The cause is that at reading bundles head2 mostly sees e tokens, whose varied contents overlap with the rate direction.

Two remedies:
- **Read through all four heads.** Heads 0 and 3 see e at reading bundles but the slots at action bundles, so a joint read could cancel the e-token contamination.
- **Accept the noise.** Even 0.36 u/h is only about 0.7 mg/dL of prediction noise, costing roughly 0.001 nats per glucose reading.

Next I'll fit a ridge read on the concatenated mixtures of all four heads, with out-of-sample R² from 5-fold cross-validation grouped by stay. If the residual sd falls well below 0.1 u/h out of sample, I'll use a multi-head value read. If it stays around 0.3, the extra heads don't help, and I'll take the single-head read and absorb its noise downstream.

```python
from sklearn.model_selection import GroupKFold
M0=mixture(0); M3=mixture(3)
Xall=np.concatenate([M0,M1,M2,M3],1); grp_s=PD.stay.values
T_comb={'ins_s':PD.ins_s.values,'ins_e':PD.ins_e.values,'ne_s':PD.ne_s.values,'ne_e':PD.ne_e.values,
        'INS':2.0*PD.ins_s.values+0.95*PD.ins_e.values,'NE':3.45*PD.ne_s.values+0.72*PD.ne_e.values}
for lam in [1e-3,1e-1]:
    for nm_,y_ in T_comb.items():
        pr_=np.zeros_like(y_)
        for tr_i,te_i in GroupKFold(5).split(Xall,y_,grp_s):
            c_,_=fit_noint(Xall[tr_i],y_[tr_i],lam); pr_[te_i]=Xall[te_i]@c_
        r_=y_-pr_; m_=y_!=0
        print('lam',lam,nm_,'CV R2 %.4f | resid sd target %.4f non-target %.4f'%(1-r_.var()/y_.var(), r_[m_].std(), r_[~m_].std()))
```