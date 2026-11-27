Decoding from the attention outputs is too noisy: the insulin Δ reaches only R² 0.63 and the gate overlaps. The cause is that the attended mixture, and which head reads u, varies by bundle.

That leaves two questions:
- **Is Δ linearly present in the raw u encoding?** If so, the z-posterior MLP, which reads u directly with no attention, can compute a clean Δ channel. It would be exactly 0 at non-drug bundles (u=0 there), so no gate would be needed.
- **Does some prelude head attend the z token with a stable weight at every bundle?** If so, that channel reaches the slots reliably.

What would confirm the route: Δ R² >0.98 on u, plus a head with near-constant attention to the z token.

```python
U=dsE['u'].numpy(); Xu=np.concatenate([U,np.ones((B,1))],1)
def r_kind(drugname,kind):
    a=at[(at.event==drugname)&(at.kind==kind)]; r=np.zeros(B); r[[ridx[x] for x in a['row'].values]]=a['rate'].values; return r
rsI=r_kind('insulin',0); reI=r_kind('insulin',2)
for nm,t_ in [('ins start',rsI),('ins end',reI),('ins delta',dI),('ne delta',dN)]:
    b_=ridge_fit(Xu[drug], t_[drug], lam=1e-4); p_=Xu@b_; on=drug
    res=p_[on]-t_[on]; print(nm,'R2 %.4f resid %.3f sd %.3f | nodrug |p| max %.3f'%(1-res.var()/t_[on].var(), res.std(), t_[on].std(), np.abs(p_[~drug]).max()))
a6=att[:,:,:4,6].numpy()
for h in range(4): print('head',h,'att->z mean %.3f p1 %.3f p99 %.3f'%(a6[:,h].mean(), np.percentile(a6[:,h],1), np.percentile(a6[:,h],99)))
```