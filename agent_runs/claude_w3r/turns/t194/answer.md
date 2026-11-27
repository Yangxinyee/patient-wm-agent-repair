Run 181 gave two key results.

**Head2 attention.** Head2 is a clean dose reader: 98.9% on the u token at action bundles. At no-action bundles it sits 92% on the observation token e. So a read vector on head2's output must give about zero for observation tokens.

**Low-variance directions.** Every stage has directions with tiny natural variation:
- x2: sd about 0.025, against a largest sd of 288.
- mean y: sd about 0.005.
- E: sd about 0.0006.
- out: sd about 0.0001.

So a modest injected dose signal along such directions would be cleanly readable and would barely disturb the LayerNorms. This supports Plan A.

Next I'll check stage 1. I'll rebuild the per-head prelude outputs from the norm1 outputs and the kv weights, and verify them against the captured attention output. Then I'll regress four targets on the head outputs at slot 0: the start rate (lag 1) and the END-atom rate (lag 2, via head1), for each drug. High R² with near-zero responses at no-action bundles would confirm stage 1. Poor R² would mean I need to gate or use a different head.

```python
Wkv=Wt['core.prelude.0.attn.kv.weight'].numpy(); Wq=Wt['core.prelude.0.attn.q.weight'].numpy(); Wo=Wt['core.prelude.0.attn.out.weight'].numpy()
print(Wkv.shape,Wq.shape,Wo.shape)
n1=P_['n1'].astype(np.float64)
Vall=np.einsum('btd,hjd->bthj',n1,Wkv[192:].reshape(4,48,192))      # b,token,head,48
HO=np.einsum('bhqt,bthj->bhqj',patt[:,:,:4,:].astype(np.float64),Vall)   # b,head,slotq,48
rec=np.einsum('bhqj,dhj->bqd',HO,Wo.reshape(192,4,48))
print('pout recon err', np.abs(rec-P_['pout'][:,:4]).max(), 'scale', np.abs(P_['pout'][:,:4]).max())
from sklearn.linear_model import LinearRegression
tg={'ins_s':PD.ins_s.values,'ne_s':PD.ne_s.values,'ins_e':PD.ins_e.values,'ne_e':PD.ne_e.values}
for hh_ in [[2],[1],[1,2],[0,1,2,3]]:
    X_=np.concatenate([HO[:,h_,0,:] for h_ in hh_],1)
    for nm_,y_ in tg.items():
        lr_=LinearRegression().fit(X_,y_); res_=y_-lr_.predict(X_)
        print('heads',hh_,nm_,'R2 %.4f resid sd %.4f | resid sd at no-action %.4f'%(lr_.score(X_,y_),res_.std(),res_[~act].std()))
```