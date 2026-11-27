The batched call fails, probably because `view_from` builds its gap features in a way that only works for one row. A loop over rows works and is cheap enough (about 11,000 small calls).

I'll compute q row by row for both models. Then I'll check the original against `Q99` and look at the size of Δq.

```python
def qrows(Xs__, kk__, dts__):
    out__=np.zeros((len(kk__),192))
    for i__,(k__,d__) in enumerate(zip(kk__,dts__)):
        out__[i__]=np.asarray(view_from(Xs__[k__][None,:], np.array([d__]))).ravel()
    return out__
Qo=qrows(X8o,kidx,dts99); Qe=qrows(X8e,kidx,dts99)
print('max err vs Q99',np.round(np.abs(Qo-Q99).max(),5),'mean err',np.round(np.abs(Qo-Q99).mean(),6))
dQ=Qe-Qo
print('|dq| mean',np.round(np.linalg.norm(dQ,axis=1).mean(),4),'| factual sd of q (sqrt total var)',np.round(np.sqrt(Qo.var(0).sum()),3),'| |q| mean',np.round(np.linalg.norm(Qo,axis=1).mean(),3))
print('code counts', {int(c):int((c99==c).sum()) for c in np.unique(c99)})
```