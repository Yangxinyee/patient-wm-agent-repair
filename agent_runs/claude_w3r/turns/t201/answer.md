Run 188 shows the trade-off is bad at both ends. Even when reading-bundle e tokens are almost ignored, the insulin target residual only falls to 0.12 u/h. The 1% of head2's attention that goes to the END-atom e token at action bundles is enough to contaminate the read once the read is unconstrained on e tokens. So no single linear value read is clean everywhere.

A gate is needed. My candidate is a pair of unused hidden units in an FFN, combining an "action present" gate with the rate signal. Two facts would make it viable:
- fitted only at action bundles, the head2 mixture reads the rate cleanly (about 0.05 u/h);
- the prelude FFN has dead hidden units.

Next I'll check both, plus the margin of a linear "action present" read on head2's mixture. The design is refuted if the action-only fit stays noisy or no dead units exist.

```python
for nm_,ycol in [('ins_s','ins_s'),('ne_s','ne_s')]:
    c_,r_=fit_noint(M2[act],PD[ycol].values[act]); m_=PD[ycol].values[act]>0
    print(nm_,'action-only fit on M2: resid sd target %.4f other-action %.4f | read on no-action sd %.2f max %.1f'%(r_[m_].std(),r_[~m_].std(),(M2[noact]@c_).std(),np.abs(M2[noact]@c_).max()))
    res_tab[('h2act',nm_)]=c_
pres=act.astype(float)*2-1
c_p,r_p=fit_noint(M2,pres); rp_=M2@c_p
print('presence read: action min %.3f mean %.3f | no-action max %.3f mean %.3f'%(rp_[act].min(),rp_[act].mean(),rp_[noact].max(),rp_[noact].mean()))
# prelude FFN hidden activity on slot tokens
W1p=Wt['core.prelude.0.ffn.0.weight'].numpy(); b1p=Wt['core.prelude.0.ffn.0.bias'].numpy(); print('prelude ffn W1',W1p.shape)
n3in=None
x1s=P_['tok0'][:,:4,:].astype(np.float64)+P_['pout'][:,:4,:].astype(np.float64)
def LNnp(x,name):
    w_=Wt[name+'.weight'].numpy(); b_=Wt[name+'.bias'].numpy(); mu=x.mean(-1,keepdims=True); v=x.var(-1,keepdims=True); return (x-mu)/np.sqrt(v+1e-5)*w_+b_
n3=LNnp(x1s,'core.prelude.0.norm3')
hid_p=n3@W1p.T+b1p; mxp=hid_p.max((0,1))
print('prelude ffn units with max preact < -2:', (mxp<-2).sum(), ' <0:',(mxp<0).sum(), ' of',len(mxp), 'min of max', mxp.min().round(2))
```