Run 144 shows that none of the helper sources can be recovered: they were defined in a way `inspect` cannot read. The data structures are intact, and their columns tell me what they hold:
- `RD`/`RD2` are per-reading tables. They carry the state row before each reading, the last value, the original prediction (`pred0`), `nll0`, insulin, NE and insulin dose-surprise columns at lags 1 to 3, and the residual.
- `HI` holds the per-reading head inputs: transition logits, bin centres and the observed bin.

Next I will:
1. Record the recovered facts in the notebook.
2. Re-derive the value-head computation in plain numpy from `WC0`, checking it against `Hh0`/`A0`/`P0`, the predicted mean (`pred0`) and the NLL (`nll0`).

I need an exact offline head before I can design and check the edit. If the recomputed probabilities match `P0` to float precision and the mean matches `pred0`, the offline head is confirmed. A mismatch would mean `WC0` or `P0` was computed differently, for example with a different z sample.

```python
with open('/work/notes.md','a') as f_:
    f_.write("""
## Restart 5 (runs 130-144) recovered
- fwd5-7 = states original on selT (300 train stays, /work/selT.json). XALL(17848,992)/STALL(stay_id,minute,has_ins,has_ne,row,prev_row,prev_min) = all bundles of 400 stays (selS+selT=stays400).
- RD/RD2: 19451 readings of those stays (stay_id,hour,event,observed,minute,prev_row=XALL row of state before reading,last_val,last_min,pred0,nll0; RD2 adds ins_k,ne_k,eps_ins_k,ins_exp_k k=1..3, res).
- HI = offline head inputs per reading (ctx,zmu,zls,reg,trans,ob,cen,ev); WC0=with_code (19451,192); Hh0/A0 hidden; P0 probs.
- Run142 lin_sep on WC: glucose vs map separable (r2 1.0); ins-present among glucose r2 .996; joint gate not linear (.78); NE among map .98.
- Helper sources NOT retrievable (inspect fails): rewrite helpers when needed.
- Plan: per drug rank-1 W1 edit: W1 += c (W1 d) w^T, b1 -= c (W1 d)(w.mu); d = WC direction moving glucose (MAP) mean but not other code; w = centred decoder of dose signal from WC; calibrate c with dose_edit both on sel1.
""")
import numpy as np
from scipy.special import erf as _erf
print([n for n in NAMES if 'attr_heads' in n])
W1_=load('attr_heads.value_bin.0.weight'); b1_=load('attr_heads.value_bin.0.bias')
W2_=load('attr_heads.value_bin.2.weight'); b2_=load('attr_heads.value_bin.2.bias')
def gelu_np(x): return 0.5*x*(1+_erf(x/np.sqrt(2)))
pre_=WC0@W1_.T+b1_; hid_=gelu_np(pre_)
print('Hh0~pre',np.abs(pre_-Hh0).max(),'Hh0~hid',np.abs(hid_-Hh0).max(),'A0~pre',np.abs(pre_-A0).max(),'A0~hid',np.abs(hid_-A0).max())
lg_=hid_@W2_.T+b2_+HI['trans']; lg_-=lg_.max(1,keepdims=True); P_=np.exp(lg_); P_/=P_.sum(1,keepdims=True)
print('P err',np.abs(P_-P0).max())
cen_=HI['cen']; m1_=(P0[:,1:]*cen_).sum(1)
print('pred0 vs sum P[1:]cen: maxabs',np.abs(m1_-RD['pred0'].values).max(), 'P0[:,0] max',P0[:,0].max())
ob_=HI['ob'].astype(int); nl_=-np.log(P0[np.arange(len(ob_)),ob_])
print('nll0 - (-logP[ob]) mean/sd', (RD['nll0'].values-nl_).mean().round(4), (RD['nll0'].values-nl_).std().round(4))
print({k:(list(v.keys())[:12] if isinstance(v,dict) else str(v)[:60]) for k,v in VOCAB.items()})
print(RD['event'].value_counts().to_dict())
```