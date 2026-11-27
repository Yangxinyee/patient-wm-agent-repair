The bigger memory works on the write side. The write plus plumbing costs −0.0006 on sel2 and +0.0002 on dsE. The head signal now follows the ideal dose memory closely (slope 3.65, correlation 0.965, sd 4.3). The switch still separates cleanly (≥1593 vs ≤−178).

The compensation fit is still too good to be genuine (R² 0.999). With a large memory, layer norm leaks it into the other dims along more than the one direction I removed. Its σ also changes, and that effect is quadratic. The clean approach is to fit the compensation on the original model's inputs, which contain only what the model actually knows. I'll then make it blind to the leakage by constraining it to be orthogonal to the leak directions, estimated by regressing the edited-minus-original difference on s and s². This run also gets the glucose_finger bin centres and the average predictive variance at glucose_finger readings, to convert the readout gain into mg/dL. I expect a genuine R² somewhere around 0.5–0.9 and a leak coefficient near zero after the constraint.

```python
WCo_=WCo; Xo=WCo_[isg][:,oth]; D8=(WC8-WCo_)[isg][:,oth]
B_=np.linalg.lstsq(np.c_[sg8,sg8**2,np.ones_like(sg8)],D8,rcond=None)[0]; A1,A2=B_[0],B_[1]
rD8=D8-np.c_[sg8,sg8**2,np.ones_like(sg8)]@B_; print('leak fit: D8 var explained %.3f'%(1-rD8.var()/D8.var()))
Qb,_=np.linalg.qr(np.c_[A1,A2]); proj=lambda X: X-(X@Qb)@Qb.T
rgg=Ridge(alpha=1.0).fit(proj(Xo),sg8); bet=rgg.coef_-Qb@(Qb.T@rgg.coef_); b0g=float(rgg.intercept_)
pg=Xo@bet+b0g; rg_=sg8-pg; print('genuine (orig wc, constrained) R2 %.3f resid sd %.3f'%(1-rg_.var()/sg8.var(), rg_.std()))
pe=WC8[isg][:,oth]@bet+b0g; lk=np.polyfit(sg8,pe-pg,1); print('leak in edited: d(beta.wc)/ds %.4f, rms diff %.4f'%(lk[0], np.std(pe-pg)))
vbc=VOCAB['value_bin_centres']; cg=None
if isinstance(vbc,dict): cg=np.asarray(vbc.get('glucose_finger',vbc.get('6')),dtype=float)
elif isinstance(vbc,list): cg=np.asarray([x for x in vbc if (isinstance(x,dict) and x.get('code')=='glucose_finger')][0].get('centres'),dtype=float) if isinstance(vbc[0],dict) else np.asarray(vbc,dtype=float)
print('centres', None if cg is None else np.round(cg,0))
pr0=tf_eval(PO,ds_s2,False)[1].numpy(); gg=code2==6; pb=pr0[gg][:,1:]/pr0[gg][:,1:].sum(1,keepdims=True)
mu_=pb@cg; var_=pb@(cg**2)-mu_**2; print('glucose_finger pred sd: mean %.1f median %.1f; mean var %.0f'%(np.sqrt(var_).mean(), np.median(np.sqrt(var_)), var_.mean()))
```