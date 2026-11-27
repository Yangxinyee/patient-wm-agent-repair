Edited q decodes the lagged rates no better than original q. For norepinephrine at lag 1 (α = 10⁻³) the R² is 0.53 edited vs 0.57 original. 98.7% of the write's effect on q lies in q's high-variance subspace. The dynamics map the z write into the network's natural directions, where it is masked. So "find a quiet direction" does not work without re-engineering several layers.

This points to a different use of z. Since z writes land in the natural, head-relevant subspace anyway, write the dose innovation along the z directions the model itself uses to move its glucose or MAP predictions. Its own dynamics and value head would then carry the effect, in-distribution, in the strongest possible way. I call this a "latent hijack".

It also reframes the norepinephrine problem. The model's errors should track the part of the dose its view does not encode, D − E[D | view]. So innovations relative to the ctx decoder (R² 0.59) may be the right signal rather than a flaw. I will check that later by regressing factual errors on them.

The first quantitative check uses fwd8. I compute the predicted glucose and MAP means from both state sets with my value-head rebuild, and regress the change on lagged rates. That gives the predicted-mean response per unit of z write. If a quiet-dim write of about 1.8 per unit of norepinephrine moves predicted MAP by only a few tenths of a mmHg, I know what transmission I need from active dimensions. I first verify the mean rebuild against the stored per-reading means.

```python
import inspect
print('mean_of', inspect.signature(mean_of), ' head_means', inspect.signature(head_means))
base_wc=WC99-Q99   # non-q part of with_code per reading (code emb + registers)
def means_from(WC,TR,CEN):
    H=WC@W1.T+b1; A_=0.5*H*(1+np.tanh(np.sqrt(2/np.pi)*(H+0.044715*H**3)))
    lg=A_@W2.T+b2+TR; lg=lg-lg.max(1,keepdims=True); p=np.exp(lg); p/=p.sum(1,keepdims=True)
    pv=p[:,1:]; return (pv*CEN).sum(1)/pv.sum(1), p
mo,po=means_from(WC99,TR99,CENm)
print('W1',W1.shape,'check vs mean_rec: MAE %.3f'%np.abs(mo-R['mean_rec'].values).mean())
mo8,_=means_from(Qo8+base_wc,TR99,CENm); me8,_=means_from(Qe8+base_wc,TR99,CENm)
dm8=me8-mo8
for grp,msk in [('gluc',mG),('map',mM)]:
    B_,*_=np.linalg.lstsq(XL[msk],dm8[msk],rcond=None)
    print(grp,'dmean sd %.3f'%dm8[msk].std(),' coef:',' '.join('%s %.3f'%(c,b) for c,b in zip(cols_,B_)))
```