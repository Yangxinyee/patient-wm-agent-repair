The start rate is almost perfectly linear in the raw u encoding (R² 0.998), and the end rate is not (R² 0.46). So a "running-rate" memory built from (start − end) is not available. No prelude head attends the z token, which rules out the z-posterior channel. The remaining source is the attention copy of u.

So the memory will be a decaying sum of start rates, which works like a recent-insulin variable. The minimal version stages the decoded start rate straight from the attention output into a cut dimension, with no gate. That works only if one decoder can be accurate where u is attended and close to zero everywhere else.

I'll fit ridge decoders with an off-target penalty, on head 3 alone and on all heads. What would confirm the 3-dimension, gate-free design:
- on-target error ≲0.1;
- leakage elsewhere p99 ≲0.1, max ≲1.

```python
clean3=drug&(a3u[:,0]>0.99)
for nm,XX,pos in [('o3',Xo,clean3),('oall',Xa,drug)]:
    for woff in [1.,10.,100.]:
        w=np.where(rep4(pos),1.,woff*rep4(pos).sum()/max(1,(~rep4(pos)).sum()))
        t_=np.where(rep4(pos),rep4(rsI),0.)
        b_=ridge_fit(XX,t_,w=w,lam=1e-4); p_=(XX@b_).reshape(B,4)
        on=pos&(rsI>0); res=p_[on]-rsI[on][:,None]; off=np.abs(p_[~pos])
        print(nm,woff,'on resid %.3f | off mean %.3f p99 %.3f max %.2f | NEonly-clean p99 %.3f'%(res.std(), off.mean(), np.percentile(off,99), off.max(), np.percentile(np.abs(p_[pos&(rsI==0)]),99)))
```