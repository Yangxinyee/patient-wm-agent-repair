The error grows with the number of readings in the bundle: MAE 1.9 for one reading, 3.5 for two, 4.0 for three and 5.2 for four. The tool's NLL is also lower than mine, 1.549 against 1.634. Class 0 has zero probability, so renormalising is not the issue.

So the tool's predictions use information from the target bundle itself. The simplest explanation is that the factual pass reads with a z drawn using the next bundle's contents. The saved states hold the posterior mean z after every bundle.

I will test the view built with bundle k+1's posterior z (the `zpost` slice of state k+1) in place of the prior. If the MAE collapses toward zero, that settles it. If it does not, the conditioning works differently, for example sequentially within the bundle.

```python
k1=kv+1
assert (S2['stay_id'][k1]==F.stay_id.values).all()
zpost_next=Xs[k1,960:992]; hrn=S2['hour'][k1]; print("next-bundle hour == reading hour:",np.mean(np.abs(hrn-F.hour.values)<1e-6).round(3))
Vp=CTX+L(zpost_next,'z_head'); pp=head_p(F,Vp,1,R,lb,code_ids); mp=(pp[:,1:]*cen).sum(1)
nllp=-np.log(pp[np.arange(len(F)),bo])
print("posterior-z: MAE %.3f | NLL mine %.4f vs tool %.4f | corr nll %.3f"%(np.abs(mp-tool).mean(),nllp.mean(),F.nll.mean(),np.corrcoef(nllp,F.nll)[0,1]))
e2=np.abs(mp-tool); print("MAE by bundle size:",{int(k):round(e2[(F.nb==k).values].mean(),3) for k in sorted(F.nb.unique())})
```