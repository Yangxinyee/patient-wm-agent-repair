Run 182 shows that reading the rates from head2's existing 48-dim attention output is too noisy. R² is only 0.80 for the insulin start rate, and residuals at no-action bundles are 0.19 u/h, since head2 reads the e token there. The value projection discards part of the rate signal that is in u.

So stage 1 has to add a new value-read direction, a rank-1 edit of the head's value weights, that reads the rate straight from the normalised token. That works only if one linear probe on the normalised tokens can read:
- the start rate from u tokens while giving about 0 for e tokens;
- the END-atom rate from e tokens while giving about 0 for reading tokens.

Next I'll test this by regression on the stacked u and e tokens of all bundles. R² ≥ 0.99 with tiny residuals on non-target tokens would confirm the value-read design. If not, I'll need a gated (nonlinear) route instead.

```python
n1u=n1[:,5,:]; n1e=n1[:,4,:]
isI=PD.has_ins_s.values>0; isN=PD.has_ne_s.values>0; isIe=PD.has_ins_e.values>0; isNe=PD.has_ne_e.values>0
Xst=np.concatenate([n1u,n1e]); one=np.ones(len(Xst))
def fitprobe(X,y,w=None,lam=1e-6):
    Xa=np.concatenate([X,np.ones((len(X),1))],1)
    A=Xa.T@(Xa*(w[:,None] if w is not None else 1))+lam*np.eye(Xa.shape[1]); b=Xa.T@((w if w is not None else 1)*y)
    c=np.linalg.solve(A,b); return c, y-Xa@c
for nm_,yu,ye in [('ins_s',PD.ins_s.values,0*PD.ins_s.values),('ne_s',PD.ne_s.values,0*PD.ne_s.values),
                  ('ins_e',0*PD.ins_e.values,PD.ins_e.values),('ne_e',0*PD.ne_e.values,PD.ne_e.values)]:
    y_=np.concatenate([yu,ye]); c_,res_=fitprobe(Xst,y_)
    ru_,re_=res_[:len(n1u)],res_[len(n1u):]
    tgt=(yu if nm_.endswith('_s') else ye); m_=tgt>0
    print(nm_,'R2 %.5f'%(1-res_.var()/y_.var()),'resid sd: on target tokens %.4f'%( (ru_ if nm_.endswith('_s') else re_)[m_].std()),
          '| u-tok no-drug %.4f'%ru_[~m_ if nm_.endswith('_s') else slice(None)].std(),'| e-tok %.4f'%re_[(~m_) if nm_.endswith('_e') else slice(None)].std(),
          '| max abs off-target %.3f'%np.abs(np.concatenate([ru_[~m_] if nm_.endswith('_s') else ru_, re_[~m_] if nm_.endswith('_e') else re_])).max())
```