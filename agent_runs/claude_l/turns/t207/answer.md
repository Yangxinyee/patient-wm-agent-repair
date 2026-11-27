The three `res_tf` variants give useful diagnostics:

- **Copy test (`dup`):** making the z token an exact copy of u changes the factual value NLL by only −0.00006. Per code the change stays within ±0.0025, and predicted means shift with sd 0.5. The copy is not perfectly neutral, because the extra token changes the attention balance. It is still far inside the 0.005 budget.
- **`ki4`:** shifts predictions by about a third as much as `dup`, with no NLL cost.
- **`ki8`:** identical to the baseline. That run had no effect at all; it was probably a no-op or a reproducibility check.

So the "copy u into the z token" mechanism is safe for the factual fit.

The open question is whether a multi-hour dose memory can be kept alive by re-injecting it through the z token at every bundle. The model's own state forgets within about an hour. The true insulin and NE effects last about 6–12 hours.

Earlier runs stored per-bundle propagation Jacobians (`Js`, 42 of them), a natural-variation covariance of the context (`Sv`), and a candidate injection/decoder pair (`DB1`, `DEC1`). Next I'll compute, for that pair:

- the per-bundle loop gain (decoder · J · injection): its mean and spread;
- the natural "phantom" signal the decoder sees without any injection: the sd of DEC1·ctx;
- the same quantities for the best generalized-eigen direction.

What the numbers would mean:
- **Feasible:** gain spread under about 5% and phantom sd small next to a moderate injection.
- **Infeasible:** large gain spread or phantom. I'd then fall back to a lag-1-only correction with its gain calibrated on the main readout.

```python
g1=np.array([DEC1@Js[i]@DB1 for i in range(len(Js))])
print('loop gain DEC1/DB1: mean %.3f sd %.3f min %.3f max %.3f'%(g1.mean(),g1.std(),g1.min(),g1.max()))
print('phantom sd (DEC1 ctx natural): %.3f ; |DB1| %.2f |DEC1| %.3f'%(np.sqrt(DEC1@Sv@DEC1),np.linalg.norm(DB1),np.linalg.norm(DEC1)))
# best generalized direction: maximise (d'J b)^2 / d'Sv d over d for each b -> b' Jm' Sinv Jm b
Mq_=Jm.T@Sinv@Jm; ev,evec=np.linalg.eigh((Mq_+Mq_.T)/2)
for j in [1,2,3,5,10]:
    b=evec[:,-j]; d=Sinv@Jm@b; d=d/(d@Jm@b)
    g=np.array([d@Js[i]@b for i in range(len(Js))])
    print('eig%d snr %.2f gain mean %.3f sd %.3f phantom %.3f'%(j,np.sqrt(ev[-j]),g.mean(),g.std(),np.sqrt(d@Sv@d)))
print('sig shape',sig.shape,'sig norms', np.round(np.linalg.norm(sig,axis=1)[:6],3))
```