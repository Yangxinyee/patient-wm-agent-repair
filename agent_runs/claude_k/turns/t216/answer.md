The value-head weights in the kernel (W1, b1, W2, b2) are torch tensors, not numpy arrays, so the matrix product failed. I will convert them and rerun the same check and analysis:
1. Verify my value-head mean rebuild against the stored per-reading means.
2. Measure how the fwd8 z18/z28 writes moved the predicted glucose and MAP means, per unit of lagged rate.

That tells me how much transmission a z-direction write must achieve to reach a lag-1 effect of about 1.5 mg/dL per unit of insulin and about 3 mmHg per unit of norepinephrine.

```python
W1n,b1n,W2n,b2n=[np.asarray(t.detach().cpu().numpy() if hasattr(t,'detach') else t,np.float64) for t in (W1,b1,W2,b2)]
def means_from(WC,TR,CEN):
    H=WC@W1n.T+b1n; A_=0.5*H*(1+np.tanh(np.sqrt(2/np.pi)*(H+0.044715*H**3)))
    lg=A_@W2n.T+b2n+TR; lg=lg-lg.max(1,keepdims=True); p=np.exp(lg); p/=p.sum(1,keepdims=True)
    pv=p[:,1:]; return (pv*CEN).sum(1)/pv.sum(1), p
base_wc=WC99-Q99
mo,po=means_from(WC99,TR99,CENm)
print('W1',W1n.shape,'check vs mean_rec: MAE %.4f'%np.abs(mo-R['mean_rec'].values).mean())
H_=WC99@W1n.T+b1n; from scipy.special import erf
A_exact=0.5*H_*(1+erf(H_/np.sqrt(2)))
print('pre-act check vs PRE: %.2e'%np.abs(H_-PRE).max())
mo8,_=means_from(Qo8+base_wc,TR99,CENm); me8,_=means_from(Qe8+base_wc,TR99,CENm)
dm8=me8-mo8
for grp,msk in [('gluc',mG),('map',mM)]:
    B_,*_=np.linalg.lstsq(XL[msk],dm8[msk],rcond=None)
    print(grp,'dmean sd %.3f'%dm8[msk].std(),' coef:',' '.join('%s %.3f'%(c,b) for c,b in zip(cols_,B_)))
```