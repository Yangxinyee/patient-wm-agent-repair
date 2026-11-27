About 20 prelude-FFN units have negligible output impact (≤0.025, against slot tokens of norm about 800; their max pre-activation is about 0). I can repurpose them for the gated decoder without measurably changing the original computation.

Next, a decoder that stays accurate at the edited rates. I fit linear reads of r, √r and r² from n1(u) on UX, which holds real action tokens under all four dose edits. A small random-GELU layer with least-squares output weights then maps those reads to r. I check accuracy by rate range (insulin up to about 19, norepinephrine up to 2), and how large the reads get on non-u tokens, which sets the gate margin. A residual of ≤0.15 that is uniform across rate ranges would confirm the approach.

```python
N1X=np.array([np.asarray(n1_of(UX[i])) for i in range(len(UX))]); yi=np.asarray(YXi,float); yn=np.asarray(YXn,float)
Xa=np.hstack([N1X,np.ones((len(N1X),1))])
def lsr(X,y,lam=1e-6): return np.linalg.solve(X.T@X/len(X)+lam*np.eye(X.shape[1]),X.T@y/len(X))
Xn1=np.hstack([NEG,np.ones((len(NEG),1))])
DEC={}
for nm,y in [('ins',yi),('ne',yn)]:
    m=y>0; T=np.stack([y,np.sqrt(y),y**2],1)
    Wr=np.stack([lsr(Xa,T[:,c]) for c in range(3)],1); Rd=Xa@Wr; Rn=Xn1@Wr
    rng=np.random.default_rng(0); K=8; C=rng.normal(size=(3,K))/np.maximum(Rd[m].std(0),1e-6)[:,None]; bq=-(Rd[m].mean(0)@C)+rng.normal(size=K)*1.0
    H=Fnn.gelu(torch.tensor(Rd@C+bq)).numpy(); Hh=np.hstack([H,np.ones((len(H),1))]); o=np.linalg.lstsq(Hh[m|(~m)],y,rcond=None)[0]; pr=Hh@o
    DEC[nm]=(Wr,C,bq,o)
    for lo,hi in [(0,1e-9),(1e-9,.3),(.3,1),(1,2),(2,5),(5,10),(10,30)]:
        s=(y>=lo)&(y<hi) if hi>1e-9 else (y==0)
        if s.sum()>5: print(nm,f'[{lo},{hi})',s.sum(),'lin-r res',round(float(np.std(Rd[s,0]-y[s])),3),'dec res',round(float(np.std(pr[s]-y[s])),3),'dec bias',round(float(np.mean(pr[s]-y[s])),3))
    print(nm,'read garbage on NEG max |R|', np.round(np.abs(Rn).max(0),1), 'sd', np.round(Rn.std(0),2))
```