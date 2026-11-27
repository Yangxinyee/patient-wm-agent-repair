Run 187 confirms a real conflict. No linear read on the normalised tokens both reads the insulin start rate cleanly from u tokens and stays near zero on reading-bundle e tokens. The best compromise leaves 0.36 u/h on target bundles and 0.20 on reading bundles. Fitting on u tokens alone gives 0.05.

So it's a trade-off between clean action bundles and noisy reading bundles. The reading-bundle noise could be gated off later, or made harmless by injecting into fast-decaying state dimensions. To choose, I'll trace the trade-off curve, down-weighting the reading-bundle e tokens by factors from 1 to 0.0001. I'll record the residuals on target, other-action and reading bundles for both insulin and norepinephrine.

A good region would show target residual ≲ 0.1 while reading-bundle residual stays ≲ 1. If the reading-bundle residual explodes as soon as the target becomes clean, a gating mechanism is required.

```python
for nm_,ycol in [('ins_s','ins_s'),('ne_s','ne_s')]:
    for lam_e in [1,0.1,0.01,0.001,0.0001]:
        X_=np.concatenate([n1u[act],n1e[noact],n1u[noact]]); y_=np.concatenate([PD[ycol].values[act],np.zeros(noact.sum()),np.zeros(noact.sum())])
        w_=np.concatenate([att2[act,5],lam_e*att2[noact,4],att2[noact,5]])
        c_=wfit(X_,y_,w_)
        rd_=M2@c_; r_=PD[ycol].values-rd_; m_=PD[ycol].values>0
        print(nm_,'lam_e',lam_e,'resid sd: target %.4f | other action %.4f | no-action %.4f (max %.2f) | e-tok read sd %.3f'%(r_[m_].std(),r_[act&~m_].std(),r_[noact].std(),np.abs(r_[noact]).max(),(n1e[noact]@c_).std()))
```