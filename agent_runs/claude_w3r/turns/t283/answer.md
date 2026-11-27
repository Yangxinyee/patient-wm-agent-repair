The gated decoder works but is imperfect. On UX, which includes the edited rates:
- Insulin residual is about 0.2 below rate 2, 0.47 at 2–5 and 0.78 at 5–10.
- Norepinephrine residual is 0.16–0.24, with a −0.47 bias at the doubled maximum (2.0).
- Without a non-u-token constraint, the reads produce huge garbage on non-u tokens (sd ~150).

The garbage matters because the injection into slot tokens has to be big enough to read cleanly through norm3. The trade-off depends on how quiet the cleanest directions are. So this run computes the natural variance spectrum of the norm3 output (the FFN read side) and of E (the b_in read side). It also scans how the non-u-token constraint trades decoder accuracy against garbage. The design is feasible if the norm3 minimum sd is ≤0.001 and the garbage can be bounded to about 20 at decoder residual ≤0.3.

```python
N3a=np.concatenate(N3,0); EEa=np.concatenate(EE,0)
for nm,Z in [('n3',N3a),('E',EEa)]:
    ev_=np.linalg.eigvalsh(np.cov(Z.T)); print(nm,'sd smallest', np.round(np.sqrt(np.maximum(ev_[:8],0)),5), 'median', round(float(np.sqrt(np.median(ev_))),4))
def dec_fit(y,lamn,K=8,seed=0):
    m=y>0; T=np.stack([y,np.sqrt(y),y**2],1)
    A=Xa.T@Xa/len(Xa)+lamn*Xn1.T@Xn1/len(Xn1)+1e-6*np.eye(193)
    Wr=np.linalg.solve(A,Xa.T@T/len(Xa)); Rd=Xa@Wr; Rn=Xn1@Wr
    rng=np.random.default_rng(seed); C=rng.normal(size=(3,K))/np.maximum(Rd[m].std(0),1e-6)[:,None]; bq=-(Rd[m].mean(0)@C)+rng.normal(size=K)
    H=np.hstack([Fnn.gelu(torch.tensor(Rd@C+bq)).numpy(),np.ones((len(Rd),1))]); o=np.linalg.lstsq(H,y,rcond=None)[0]
    return Wr,C,bq,o,H@o,Rn
for nm,y in [('ins',yi),('ne',yn)]:
    for lamn in [1e-3,1e-2,1e-1]:
        Wr,C,bq,o,pr,Rn=dec_fit(y,lamn)
        rs=[round(float(np.std(pr[(y>=lo)&(y<hi)]-y[(y>=lo)&(y<hi)])),2) for lo,hi in [(.3,1),(1,2),(2,5),(5,30)] if ((y>=lo)&(y<hi)).sum()>5]
        print(nm,lamn,'res by range',rs,'res none',round(float(np.std(pr[y==0])),2),'garbage max',np.round(np.abs(Rn).max(0),1))
```