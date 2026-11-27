Head 3's 48-dim value projection simply does not carry the start rate well: even on clean bundles the error is 0.33–0.5 (rate sd about 1.1). The rate is dropped at the value projection, which also explains why the model responds so weakly to dose changes.

Better design: repurpose one value dimension of head 3 to carry a rate decoder fitted on the full norm1 input, where the rate is near-perfectly linear. I zero that dimension's original output columns and route it only to the staging dimension. The slot then receives the attention-weighted decoder output: the rate where head 3 attends u, and leakage elsewhere.

I'll fit this 192-dim decoder on the attention-weighted norm1 features, with an off-target penalty. I'll also rank head 3's value dimensions by how much they contribute, to pick the cheapest one to repurpose.

What would confirm the gate-free design: on-target error ≲0.1 with leakage p99 ≲0.1.

```python
n1=pr['n1'].numpy(); A3=att[:,3,:4,:].numpy()          # B,4,7
Fm=np.einsum('bst,btd->bsd',A3,n1)                       # head-3 attention-weighted LN1 features
Xf=np.concatenate([Fm.reshape(-1,192),np.ones((B*4,1))],1)
pos=clean3
for woff in [1.,10.,100.,1000.]:
    w=np.where(rep4(pos),1.,woff*rep4(pos).sum()/max(1,(~rep4(pos)).sum()))
    t_=np.where(rep4(pos),rep4(rsI),0.)
    b_=ridge_fit(Xf,t_,w=w,lam=1e-4); p_=(Xf@b_).reshape(B,4)
    on=pos&(rsI>0); res=p_[on]-rsI[on][:,None]; off=np.abs(p_[~pos])
    print(woff,'on resid %.3f | off mean %.4f p99 %.3f max %.2f | NEonly p99 %.3f | 1atom-start mean %.3f'%(res.std(), off.mean(), np.percentile(off,99), off.max(), np.percentile(np.abs(p_[pos&(rsI==0)]),99), p_[ins&~clean3&(rsI>0)].mean()))
Wo=PK['core.prelude.0.attn.out.weight'].numpy()
imp=o3.reshape(-1,48).std(0)*np.linalg.norm(Wo[:,144:192],axis=0)
print('head3 vdim importance, 6 smallest:', [(int(k),round(float(imp[k]),3)) for k in np.argsort(imp)[:6]], 'median %.3f'%np.median(imp))
```