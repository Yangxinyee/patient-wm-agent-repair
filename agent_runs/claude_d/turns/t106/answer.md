Run 94 shows the per-vector std of the slot mean is nearly constant at these rows (61–69). Normalization therefore cannot be what destroyed readability. The real cause is my ridge setup: normalized features (out, LayerNorm'd slots) are about 65× smaller than raw h, so a fixed α=30 over-shrank them to nothing. My earlier "out R² 0.01" and the notebook's "z posterior 0.00" may both be regularization artifacts. I'll redo the decoding with standardized features and several α. If `out` then decodes the surprise about as well as h, the value head already sees the dose signal and just weights it too weakly. A head-level edit would then be viable.

```python
from sklearn.preprocessing import StandardScaler
blocks={'out':slice(0,192),'h':slice(192,960),'z':slice(960,992)}
Xs_=StandardScaler().fit_transform(X)
for off in [0,1,2]:
    ok=[i for i in src if i+off<len(X) and sid3[i+off]==sid3[i]]; rows=np.array(ok)+off
    t=T0[ok]; s=sid3[ok]; ye=np.array([RB[a][0][b]-RB[a][1][b] for a,b in zip(s,t)])
    Xbak=X; X=Xs_
    print('off',off,{nm:[r2cv(rows,ye,sl,alpha=a) for a in (3,30,300)] for nm,sl in blocks.items()})
    X=Xbak
```