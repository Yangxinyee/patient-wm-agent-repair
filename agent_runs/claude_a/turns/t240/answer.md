Within a bundle all four slots behave the same. Head 3 either fully attends the u token (89% of insulin-only bundles, 95% with both drugs, 93% norepinephrine-only) or does not attend it at all. The original and cut models agree, so this is the model's own behaviour, not damage from my cuts.

Something about the 11% of insulin bundles where head 3 ignores u must differ. Next I'll check:
- what those bundles contain;
- whether another head takes over the u token there;
- the gate and rate fits restricted to bundles where head 3 does attend u.

If another head covers the missed bundles, I can stage from both heads. If the clean subset gives R² >0.95 and a clean gate, the plan works for about 90% of doses, and I'll scale the gain to make up the rest.

```python
import collections
at=dsE['at']; bd=dsE['bd']; print(list(bd.columns)[:12], list(at.columns)[:14])
ridx={r:i for i,r in enumerate(bd['row'].values)}
bi=np.array([ridx[x] for x in at['row'].values])
low=ins&(a3u[:,0]<0.01); hi=ins&(a3u[:,0]>0.99)
for nm,mk in [('low',low),('hi',hi)]:
    c=collections.Counter(at['event'].values[mk[bi]]); n=mk.sum()
    print(nm,n,{k:round(v/n,2) for k,v in c.most_common(6)})
for h in range(4): print('head',h,'att->u low %.2f hi %.2f nodrug %.3f'%(att[low][:,h,:4,5].mean().item(), att[hi][:,h,:4,5].mean().item(), att[~drug][:,h,:4,5].mean().item()))
clean=drug&(a3u[:,0]>0.99)
bg=ridge_fit(Xo,np.where(rep4(clean),1.,-1.),lam=1e-3); gg=(Xo@bg).reshape(B,4)
print('gate clean min %.3f | nodrug max %.3f | lowatt-drug max %.3f'%(gg[clean].min(), gg[~drug].max(), gg[drug&~clean].max()))
mc=rep4(clean)
for nm,r_ in [('ins',ri),('ne',rn)]:
    b_=ridge_fit(Xo[mc], rep4(r_)[mc], lam=1e-4); s_=(Xo@b_).reshape(B,4); on=clean&(r_>0)
    res=s_[on]-r_[on][:,None]; print(nm,'clean R2 %.4f resid %.3f other-drug p99 %.3f'%(1-res.var()/r_[on].var(), res.std(), np.percentile(np.abs(s_[clean&(r_==0)]),99)))
```