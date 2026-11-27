The previous run crashed before it defined `design_glu`, so it does not exist in the kernel. I will define it and rerun the same regression. The expectations are the same: insulin lag-1 near −2 per u/h, smaller at longer lags.

```python
def design_glu(R):
    T0=np.eye(5)[tier_of(R.g0.values)]; g0=(R.g0.values-200)/50
    m1=np.isfinite(R.g1.values); g1=np.where(m1,(R.g1.values-200)/50,0); T1=np.eye(5)[tier_of(np.where(m1,R.g1.values,100))]*m1[:,None]
    ml=np.isfinite(R.gl0.values); gl=np.where(ml,(R.gl0.values-200)/50,0)
    return np.column_stack([T0,T0*g0[:,None],T0*g0[:,None]**2,T1,g1,m1,gl,ml])
Xc=design_glu(RI); Xd=RI[[f'd{l}' for l in range(1,7)]].values
XX=np.column_stack([Xc,Xd]); b=np.linalg.lstsq(XX,RI.y.values,rcond=None)[0]
res_=RI.y.values-XX@b; s2=res_.var(); cov=s2*np.linalg.pinv(XX.T@XX)
print("rows",len(RI),"resid sd",round(np.sqrt(s2),2))
print("true one-step insulin coefs lag1..6:",np.round(b[-6:],3),"se",np.round(np.sqrt(np.diag(cov))[-6:],3))
```