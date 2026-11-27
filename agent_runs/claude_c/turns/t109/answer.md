The pass-5 comparison shows two things.

1. Scaling the dose input x5 moves the state about 5 times more at norepinephrine bundles than at insulin bundles: median 0.19 against 0.035 of the natural spread, with a maximum of 0.84. So the action token can move the state a lot.
2. The induced change is not selective. Its top direction at insulin bundles (two directions carry 95% of it) has natural variation 25 times the induced change, and far more at reading bundles. Amplifying in state space would therefore amplify ordinary dynamics too. That rules out edits on shared tensors along state directions.

The four slots also show identical relative changes to three decimals, which suggests they are near-copies.

The remaining selective channels are the action encoder, which is bounded by its norms, and the z token, which is selective at drug bundles through p·z. Before spending a forward pass on whether the z token has any influence on the state, I need two things from saved data:
- whether the slots are effectively one 192-dimensional state;
- the factual file's layout, so I can match readings to state rows and compute the glucose-surprise direction in z.

If the slots are copies, the state is effectively 192-dimensional and later directions are simpler.

```python
Hs = [Xo[:, 192+192*s: 384+192*s] for s in range(4)]
for s in range(1,4):
    dif = np.linalg.norm(Hs[s]-Hs[0],axis=1)/np.linalg.norm(Hs[0],axis=1)
    cc = np.corrcoef(Hs[s].ravel(), Hs[0].ravel())[0,1]
    print('slot',s,'rel diff to slot0 median %.4f max %.4f corr %.5f'%(np.median(dif), dif.max(), cc))
print('h norm median', np.median(np.linalg.norm(Hs[0],axis=1)), 'out norm median', np.median(np.linalg.norm(Xo[:,:192],axis=1)))
import pandas as pd
fac = pd.read_csv('/work/fwd_2/factual_original.csv')
print(fac.shape, list(fac.columns)); print(fac.head(3).to_string())
print(fac['code'].value_counts().to_dict() if 'code' in fac else '')
```