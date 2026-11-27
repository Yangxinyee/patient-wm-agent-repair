The random-feature decoder is worse than plain linear: insulin error ≥ 0.12 even with 64 units. There are also no free z_post units, since every unit has weight ≥ 0.14 into the posterior-mean outputs. So a multi-unit writer would need repurposing paid for in factual cost, and it doesn't fit better anyway.

What matters for the readouts is the average shift, not per-configuration scatter. A linear decoder may be acceptable if, for each multiplier, mean[p·u(m) − p·u(1)] ≈ (m−1)·rate.

So I'll measure, for each multiplier and drug, the ratio of decoded shift to true shift, at two ridge strengths. I'll also score robustness: the remaining rebuild error (~1.2e-4 along p̂) scaled by |p| tells how much is lost in practice.

If ratios are within ~3% for x0 to x3, I'll keep a linear writer and fix the empty-set offset. If not, I'll consider adding a channel inside the action encoder.

```python
recs_=[]
for a,b,w in cnt.itertuples(index=False):
    if a>0:
        for m in [0,0.5,1.5,2,3]: recs_.append(('I',m,w,u_c(a,b,mi=m),u_c(a,b),min(a*m,11.5)-min(a,11.5)))
    if b>0:
        for m in [0,0.5,1.5,2,3]: recs_.append(('N',m,w,u_c(a,b,mn=m),u_c(a,b),b*m-b))
for al in [1e-4,3e-4,1e-3]:
    pi_=wridge(U_,YI,Wt,al); pn_=wridge(U_,YN,Wt,al)
    out_r=[]
    for dr_ in ['I','N']:
        for m in [0,0.5,1.5,2,3]:
            sel__=[r for r in recs_ if r[0]==dr_ and r[1]==m]
            w_=np.array([r[2] for r in sel__],float); pv=pi_ if dr_=='I' else pn_
            num__=np.array([pv@(r[3]-r[4]) for r in sel__]); den__=np.array([r[5] for r in sel__])
            out_r.append(f"{dr_}x{m}:{np.sum(w_*num__)/np.sum(w_*den__):.3f}")
    print('alpha',al,'|pI|',round(np.linalg.norm(pi_)),'|pN|',round(np.linalg.norm(pn_)),' '.join(out_r))
```