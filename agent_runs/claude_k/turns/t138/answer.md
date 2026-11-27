Two findings from run 124:
- **Reading table.** R99 is a pandas table of the 5,521 factual readings, with `prev_val` and `dt` among its columns. The tool's factual CSV lists the same 5,521 readings in a different order, so I'll align them on (stay, hour, event).
- **`last_numeric` has 8 inputs**, not the 4 the architecture text implies. That matches my old note that I had to fit its input layout.

Next I'll rebuild the value head explicitly from the stored `with_code` vectors (WC99): pre-activation, exact GELU, output layer plus transition bias, softmax, then mean over the 16 bin centres. I'll compare the result with the tool's predicted means and NLLs.

A mean absolute error of about 1 mg/dL or less, with NLL agreeing to about 0.01, would confirm the offline value-head model is good enough. Then I can use it to cost unit repurposing and to fit probes. A large mismatch would mean WC99 or TR99 isn't what I think, and I'd rebuild it from the states.

```python
from scipy.special import erf
def gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))
print(R99.columns.tolist())
W1=load('attr_heads.value_bin.0.weight'); b1=load('attr_heads.value_bin.0.bias')
W2=load('attr_heads.value_bin.2.weight'); b2=load('attr_heads.value_bin.2.bias')
pre=WC99@W1.T+b1
print('PRE match max', np.abs(pre-PRE).max())
Hh_=gelu(pre); lg=Hh_@W2.T+b2+TR99
pp=np.exp(lg-lg.max(1,keepdims=True)); pp/=pp.sum(1,keepdims=True)
mu=(pp[:,1:]*CENm).sum(1); mu2=mu/pp[:,1:].sum(1)
key=R99[['stay_id','hour','event']].copy(); key['mu']=mu; key['mu2']=mu2; key['p0']=pp[:,0]; key['pmall']=pmall
mm=key.merge(fa,on=['stay_id','hour','event'],how='inner'); print(len(mm))
print('MAE mu vs tool', np.abs(mm.mu-mm.predicted_mean).mean(), 'mu2', np.abs(mm.mu2-mm.predicted_mean).mean(), 'pmall vs tool', np.abs(mm.pmall-mm.predicted_mean).mean(), 'p0 mean', mm.p0.mean())
print(mm.groupby('event').apply(lambda d: pd.Series({'mae':np.abs(d.mu2-d.predicted_mean).mean(),'bias':(d.mu2-d.predicted_mean).mean()})))
```