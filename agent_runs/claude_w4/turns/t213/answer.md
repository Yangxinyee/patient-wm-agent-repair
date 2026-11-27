The direct route works. The original FFN passes δ_I added to x1 through almost unchanged (ratio about 1.0, cosine 0.96–0.99) even at very large magnitudes. Predicted effects stay at about +0.99 for finger and lab and about 0 for arterial and cuff MAP. So the dose term can be written exactly, independent of state, through head 1's value path.

One constraint remains: head 1 can only write vectors in the 48-dimensional output space of its slice of the output projection. So I'll re-solve for minimum-norm δ_I (finger and lab +1, MAP 0) and δ_N (MAP +1, glucose 0) within that space. Then I'll recheck the FFN pass-through and the spread of effects across bundles.

```python
GmW = Gm @ Wo1                        # 4x48
def mind_head(target):
    a = GmW.T @ np.linalg.solve(GmW @ GmW.T, target); return a, Wo1 @ a
aI_h, dI_h = mind_head(np.array([1.,1.,0.,0.])); aN_h, dN_h = mind_head(np.array([0.,0.,1.,1.]))
print('|delta_I| head-range %.2f (free %.2f)  |delta_N| %.2f (free %.2f)' % (np.linalg.norm(dI_h), np.linalg.norm(dI), np.linalg.norm(dN_h), np.linalg.norm(dN)))
for nm_, d_ in [('I', dI_h), ('N', dN_h)]:
    dt_ = torch.tensor(d_, dtype=torch.float32)
    for sc in [3.0, 10.0]:
        dx2 = (ffn_block(x1s0 + sc*dt_) - base_x2)/sc
        msk = iI if nm_=='I' else iN
        resp = dx2[msk].reshape(-1,192).numpy()
        pe = Gm @ resp.mean(0)
        pb = np.einsum('ki,nij,j->nk', Wst, Jm_, resp.reshape(-1,4,192).mean(1)[:300] if False else d_) if False else None
        print(nm_, 'scale', sc, '|dx2|/|d| %.3f' % (np.linalg.norm(resp.mean(0))/np.linalg.norm(d_)), 'pred effects/unit', np.round(pe,3))
pbI = np.einsum('ki,nij,j->nk', Wst, Jm_, dI_h); pbN = np.einsum('ki,nij,j->nk', Wst, Jm_, dN_h)
print('per-bundle (J) I mean', np.round(pbI.mean(0),3), 'sd', np.round(pbI.std(0),3), '| N mean', np.round(pbN.mean(0),3), 'sd', np.round(pbN.std(0),3))
```