fbin(event, value) maps a value to its fine bin. Next I will compute replica head logits for all Xf rows, logits = W2·gelu(W1·x + b1) + b2 + transition[code·17 + last bin], and the NLL of the observed fine bin per code. This checks that the replica head plus Xf reproduces sensible factual NLLs (tool setA: finger 1.53, lab 1.26, arterial 1.35, cuff 1.75). If it does, I can use it as the factual-cost evaluator.

```python
from scipy.special import erf, log_softmax
GID=np.array([6,10,7,5])
def gel(x): return 0.5*x*(1+erf(x/np.sqrt(2)))
def head_logits(X,c4,lb,W1_=None,b1_=None,W2_=None,b2_=None,TR_=None):
    W1_=W1 if W1_ is None else W1_; b1_=b1 if b1_ is None else b1_; W2_=W2 if W2_ is None else W2_; b2_=b2 if b2_ is None else b2_; TR_=TR if TR_ is None else TR_
    H=gel(X@W1_.T+b1_); return H@W2_.T+b2_+TR_[GID[c4]*17+lb], H
obf=np.array([fbin(CODES4[c],v) for c,v in zip(cf,uf)]).astype(int)
L0,H0=head_logits(Xf.astype(np.float64),cf,lbf)
LS0=log_softmax(L0,axis=1); nll0=-LS0[np.arange(len(obf)),obf]
print('obf range',obf.min(),obf.max())
for c in range(4): print(CODES4[c], (cf==c).sum(), round(nll0[cf==c].mean(),3))
```