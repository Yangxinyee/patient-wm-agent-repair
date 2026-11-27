The two insulin "kinds" are start and end atoms. A typical insulin bundle ends the previous infusion (kind 2, carrying the old rate) and starts a new one (kind 0, new rate). Single-atom bundles are a start with nothing running, or a stop with no restart. `bundle_rates` kept the end atom's old rate, so my decoder target was wrong, which explains R² 0.23.

This leads to a cleaner design. The memory should accumulate Δ = (start rate − end rate) at each insulin bundle, which adds up to the currently running rate. It needs no decay, and the dose edit scales every atom consistently.

Head 3 covers two-atom bundles and head 1 covers single-atom ones, so I'll decode from all four heads' outputs together. That is exactly what a new row of the attention output matrix can read. Next I'll fit a drug gate and Δ decoders for insulin and norepinephrine on that 192-dim output, and check whether glucose_finger bin edges fall on the dosing tier threshold of 150.

- **Workable:** clean gate margin and Δ R² >0.97 with small leakage at the other drug.

```python
def delta_by_bundle(drugname):
    a=at[at.event==drugname]; d=np.zeros(B)
    sgn=np.where(a['kind'].values==0,1.,np.where(a['kind'].values==2,-1.,0.))
    np.add.at(d, np.array([ridx[x] for x in a['row'].values]), sgn*a['rate'].values); return d
dI=delta_by_bundle('insulin'); dN=delta_by_bundle('norepinephrine')
print('NE kinds', at[at.event=='norepinephrine']['kind'].value_counts().to_dict())
oall=o[:,:,:4].permute(0,2,1,3).reshape(B,4,192).numpy()
Xa=np.concatenate([oall.reshape(-1,192), np.ones((B*4,1))],1)
bgA=ridge_fit(Xa,np.where(rep4(drug),1.,-1.),lam=1e-3); gA=(Xa@bgA).reshape(B,4)
print('gate drug min %.3f nodrug max %.3f'%(gA[drug].min(), gA[~drug].max()))
for nm,d_,on in [('ins',dI,ins),('ne',dN,ne)]:
    b_=ridge_fit(Xa[md], rep4(d_)[md], lam=1e-4); s_=(Xa@b_).reshape(B,4)
    res=s_[on]-d_[on][:,None]; off=drug&~on
    print(nm,'dR2 %.4f resid %.3f |d| sd %.3f other-drug p99 %.3f nodrug p99 %.2f'%(1-res.var()/d_[on].var(), res.std(), d_[on].std(), np.percentile(np.abs(s_[off]),99), np.percentile(np.abs(s_[~drug]),99)))
k=list(VOCAB.keys()); print(k[:12])
```