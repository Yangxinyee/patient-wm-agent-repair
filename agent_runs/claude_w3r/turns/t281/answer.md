Collection is fast: 3182 bundles in 1.4 s. But no prelude-FFN unit has a maximum pre-activation below −3 on slot tokens, so my notes' "154 dead units" must have used a looser criterion.

I will look at the pre-activation distribution, and at the units' output impact: column norm of ffn.2 times typical |GELU| activation. A unit whose output impact is negligible can be repurposed without changing the original computation. If there are at least 8 units with near-zero impact, the gated decoder can live there. If not, I will need a different gating location.

```python
PRA=np.array(PRE); print('max-pre quantiles', np.round(np.quantile(PREm,[0,.01,.05,.1,.2,.5]),2), 'counts <0,<-1,<-2:', (PREm<0).sum(),(PREm<-1).sum(),(PREm<-2).sum())
c2=np.linalg.norm(F2w.numpy(),axis=0)
# typical activation magnitude per unit over slot tokens
act=[]; 
for i in range(0,len(N3),8):
    pre=torch.tensor(N3[i])@F1w.T+F1b; act.append(np.abs(Fnn.gelu(pre).numpy()).max(0))
actm=np.max(np.array(act),0); imp=actm*c2
o=np.argsort(imp); print('lowest impact units', o[:20].tolist()); print('impact', np.round(imp[o[:20]],5)); print('their max-pre', np.round(PREm[o[:20]],2)); print('impact quantiles', np.round(np.quantile(imp,[.05,.2,.5]),3))
```