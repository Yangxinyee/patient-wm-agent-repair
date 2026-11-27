Run 186 shows that joint reads across all four heads barely help. The combined insulin signal still has a residual of about 0.5–0.8 mg/dL on target bundles, and the norepinephrine signal about 0.5 mmHg. That noise is tolerable but not good.

The likely culprit is that my fits demanded zero from every e token, including the END-atom e tokens of action bundles. Those carry rates almost equal to the start rate, while head2 gives them only 1% weight. So I'll refit, keeping only the tokens each head actually attends:
- head2 start-rate read: u tokens (target = start rate) plus e tokens of reading bundles only (target 0);
- head1 END-rate read: END e tokens (target = END rate), plus reading e tokens and slot tokens at reading bundles (target 0).

If the head2 residual drops to about 0.05 u/h, stage 1 is a clean single-head value edit. If not, gating is unavoidable.

```python
def wfit(X,y,w,lam=1e-6):
    A=X.T@(X*w[:,None])+lam*np.eye(X.shape[1]); c=np.linalg.solve(A,X.T@(w*y)); return c
noact=~act
att2=pattf[:,2,:4,:].mean(1); att1=pattf[:,1,:4,:].mean(1)
for nm_,ycol in [('ins_s','ins_s'),('ne_s','ne_s')]:
    X_=np.concatenate([n1u[act],n1e[noact],n1u[noact]]); y_=np.concatenate([PD[ycol].values[act],np.zeros(noact.sum()),np.zeros(noact.sum())])
    w_=np.concatenate([att2[act,5],att2[noact,4],att2[noact,5]])
    c_=wfit(X_,y_,w_); 
    rd_=M2@c_; r_=PD[ycol].values-rd_          # actual mixture read
    m_=PD[ycol].values>0
    print('head2',nm_,'mixture read resid sd: target %.4f | other action %.4f | no-action %.4f | max|no-act| %.3f |w| %.1f'%(r_[m_].std(),r_[act&~m_].std(),r_[noact].std(),np.abs(r_[noact]).max(),np.linalg.norm(c_)))
    res_tab[('h2fit',nm_)]=c_
slot_n1=n1[:,:4,:].mean(1)
for nm_,ycol in [('ins_e','ins_e'),('ne_e','ne_e')]:
    X_=np.concatenate([n1e[act],n1e[noact],slot_n1[noact],slot_n1[act]]); y_=np.concatenate([PD[ycol].values[act],np.zeros(noact.sum()),np.zeros(noact.sum()),np.zeros(act.sum())])
    w_=np.concatenate([att1[act,4],att1[noact,4],att1[noact,:4].sum(1),att1[act,:4].sum(1)])
    c_=wfit(X_,y_,w_)
    rd_=M1@c_; r_=PD[ycol].values-rd_; m_=PD[ycol].values>0
    print('head1',nm_,'mixture read resid sd: target %.4f | other action %.4f | no-action %.4f | max|no-act| %.3f |w| %.1f'%(r_[m_].std(),r_[act&~m_].std(),r_[noact].std(),np.abs(r_[noact]).max(),np.linalg.norm(c_)))
    res_tab[('h1fit',nm_)]=c_
```