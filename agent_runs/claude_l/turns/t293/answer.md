Recorded and synthetic u match to 1e-15. But the action encoding stops changing above an insulin rate of about 11.5 (the normalized input appears to be clipped at ±10). Between 5 and 11.5 it changes only slightly, and the norepinephrine encoding flattens above about 1.5. So no channel reading u can follow doses beyond that range, and the scored ×3 edits (and ×2 at high rates) will be partly out of reach.

The bigger question is gain. The u token moves a lot per dose unit, yet the prediction moves only −0.62. If the z token has a similarly weak Jacobian, writing a large effect through it would push LayerNorm into its nonlinear range. An alternative is to write only a small, decodable trace through the z token and amplify it on the read side through `z_prior → z_head` in the value head, which I have already zeroed.

Next I will print:
- the norms of the existing Jacobians in `jac.npz`,
- the `bview` code that shows how z_prior feeds the view,
- the z_prior and z_head shapes,
- the reading codes and the share of high insulin rates.

A small token-to-prediction Jacobian (≲0.3 per unit norm) favours the read-side design.

```python
import inspect, re
src=open('/work/wm2.py').read(); i=src.find('def bview'); print(src[i:i+900])
print({n:SHAPES[n] for n in NAMES if n.startswith(('z_prior','z_head','value_head.query','query_norm'))})
J=np.load('/work/jac.npz'); Js=J['Js']; nj=np.linalg.norm(Js,axis=2)
print('Js row norms median per output',np.median(nj,0).round(3),'p90',np.percentile(nj,90,0).round(3),'sk[:3]',J['sk'][:3].tolist())
print('MEASC',MEASC,'RA col1 uniq',np.unique(TT['RA'][:,1])[:8],'col3 uniq',np.unique(TT['RA'][:,3])[:20])
ri_=DAm[DAm[:,4]>0,2]; print('ins rate pct >5.75: %.3f  >3.83: %.3f  max %.1f'%((ri_>5.75).mean(),(ri_>3.83).mean(),ri_.max()))
rn_=DAm[DAm[:,5]>0,3]; print('ne rate pct >0.5: %.3f >0.75: %.3f max %.2f'%((rn_>0.5).mean(),(rn_>0.75).mean(),rn_.max()))
```