Of the variants tried, only the age units moved the glucose error, and even then only to about 6.2. Changing age units also made MAP worse, so the unit isn't the main cause. Sampling z didn't help either, so the error is systematic rather than noise from the 4 draws. That points to the prior/gap pathway: perhaps a wrong activation in `gap` or `z_prior`, or the mean/log-sd halves swapped. Below I test those, plus z set to zero, to see how sensitive the predictions are to the z term.

```python
import copy
def predict2(FR, act_gap=gelu_, act_zp=gelu_, swap=False, zero_z=False, **kw):
    global gelu_
    return None
# quick patch: re-implement just ctx/z part with options via temporary weights
def variant(tag, act=None, swap=False, zero_z=False):
    W = dict(Wd)
    if swap:
        W['z_prior.2.weight'] = np.concatenate([Wd['z_prior.2.weight'][32:],Wd['z_prior.2.weight'][:32]])
        W['z_prior.2.bias'] = np.concatenate([Wd['z_prior.2.bias'][32:],Wd['z_prior.2.bias'][:32]])
    if zero_z:
        W['z_prior.2.weight'] = np.zeros_like(Wd['z_prior.2.weight']); W['z_prior.2.bias']=np.zeros_like(Wd['z_prior.2.bias'])
    score(predict(FR,typ=1,gap_scale=10.0,W=W), tag)
variant('swap',swap=True); variant('zero_z',zero_z=True)
# activation test: temporarily replace gelu_ in gap/z_prior by monkeypatch
_g = gelu_
for nm,fa in [('relu',lambda x: np.maximum(x,0)),('silu',lambda x: x/(1+np.exp(-x))),('tanh',np.tanh)]:
    gelu_ = fa; score(predict(FR,typ=1,gap_scale=10.0), 'all_'+nm)
gelu_ = _g
```