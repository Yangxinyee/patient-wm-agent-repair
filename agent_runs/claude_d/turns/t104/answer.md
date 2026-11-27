The slot mean carries the dose surprise (R² 0.60), and every slot carries it about equally. Yet `out`, which is exactly `out_norm(slot mean)`, carries none (R² 0.01). The only thing LayerNorm removes is each vector's mean over dimensions and its overall scale. So the surprise is most likely stored in the mean or the scale of the state vector, which every LayerNorm downstream, including all the heads, discards. That would explain why the model barely acts on it.

I'll test this directly. Two features, the mean and the std of the slot mean across dimensions, should explain most of the surprise, and the LayerNorm-normalized vector should explain none.

```python
Hs=X[:,192:960].reshape(len(X),4,192); Xm=Hs.mean(1)
mu=Xm.mean(1); sd=Xm.std(1); Xln=(Xm-mu[:,None])/sd[:,None]
for off in [0,1]:
    ok=[i for i in src if i+off<len(X) and sid3[i+off]==sid3[i]]; rows=np.array(ok)+off
    t=T0[ok]; s=sid3[ok]; ye=np.array([RB[a][0][b]-RB[a][1][b] for a,b in zip(s,t)])
    Xbak=X; X=np.c_[mu,sd,Xln]
    print('off',off,'mu+sd',r2cv(rows,ye,slice(0,2),alpha=1e-3),'mu',r2cv(rows,ye,slice(0,1),alpha=1e-3),'sd',r2cv(rows,ye,slice(1,2),alpha=1e-3),'LN(meanslot)',r2cv(rows,ye,slice(2,194)))
    X=Xbak
print('corr(mu,sd) all rows',np.corrcoef(mu,sd)[0,1].round(2),'mu range',np.percentile(mu,[1,50,99]).round(3))
```