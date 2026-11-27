Two findings from run 288.

**The saved files.** The current head tensors match E4.npz exactly. The encoder (`encode_action.mlp1`) is at its original weights, so the E5 encoder units were removed.

**The factual cost, which matters more.** fwd15 and fwd16 both give edited NLL 1.52200 against original 1.51191 on the 100 sel1 tuning stays. That is +0.0101, double the 0.005 budget.
- map_arterial carries much of it (+0.019).
- Glucose_finger probably adds about +0.012; its key collided in my printout, so this is inferred.
- My offline estimate for E4 was only about +0.003, so the offline cost model is badly wrong.
- My guess at the sequence: fwd15 was E5, I then dropped the encoder, and fwd16 (E4) came out identical. That would mean the cost comes from the head channels, not the encoder.

As things stand, the submission would fail the factual condition. Fixing that is now the top priority.

Next I will locate the cost in the per-reading CSVs. For each event I will look at:
- the size of the edited-minus-original shift in the predicted mean;
- whether NLL losses sit in the rows with large shifts;
- whether the shifts point toward the observed values (slope of residual on shift).

What the slope would tell me:
- **Slope near 0**: the channel's shifts on factual rows are noise. Its gain must then shrink, or it needs a gate.
- **Slope near 1 with cost still high**: something else is changing, such as the spread of the predictive distribution.

```python
import pandas as pd, numpy as np
fe=pd.read_csv('/work/fwd_16/factual_edited.csv'); fo=pd.read_csv('/work/fwd_16/factual_original.csv')
f15=pd.read_csv('/work/fwd_15/factual_edited.csv')
print('15vs16 maxdiff', round(float(np.abs(f15.nll-fe.nll).max()),5), round(float(np.abs(f15.predicted_mean-fe.predicted_mean).max()),4), 'mean nll orig', round(fo.nll.mean(),5))
d=fo[['stay_id','hour','event','observed']].copy(); d['dn']=fe.nll-fo.nll; d['dp']=fe.predicted_mean-fo.predicted_mean
d['e0']=fo.observed-fo.predicted_mean
for ev,g in d.groupby('event'):
    big=g.dp.abs()>1
    sl=np.cov(g.e0,g.dp)[0,1]/max(g.dp.var(),1e-9)
    print(ev[:9], len(g), 'dNLL %.4f'%g.dn.mean(), 'dp mean %.2f rms %.2f'%(g.dp.mean(), np.sqrt((g.dp**2).mean())), 'f|dp|>1 %.2f'%big.mean(),
          'dNLL big %.4f small %.4f'%(g.dn[big].mean() if big.any() else 0, g.dn[~big].mean()), 'slope e0~dp %.2f'%sl)
print([(k, v.shape) for k,v in list(globals().items()) if isinstance(v,pd.DataFrame) and not k.startswith('_')][:25])
print(list(pd.read_csv('/data/tuning.csv', nrows=3).columns))
```