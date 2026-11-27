With the swap active, the context after a drug bundle is much more sensitive to the z token. The best marker direction reaches a signal-to-noise of 4.5 (insulin) and 3.8 (NE) per unit of injection. A marker written at a drug bundle can therefore be read back clearly at the next reading, even with a modest injection of about 5 units.

The catch is that for the best direction the decoded gain varies across bundles: sd 0.59 for insulin and 1.27 for NE, against a mean of 1. That would make the per-bundle effect noisy and cost factual NLL.

This supports a two-part design for a strong one-hour effect:
1. **Core side:** a swap plus a gated residual marker along a direction that is decodable and consistent across bundles.
2. **Head side:** the unused prior path (`z_prior` decodes the marker from the context; `z_head` adds it to the value head's view). It turns the decoded residual into a prediction shift along a view direction that moves finger and lab, or arterial and cuff, together.

Next I'll pick the marker directions by trading signal-to-noise against gain consistency. For candidate directions (top generalized eigenvectors, top singular vectors of the mean Jacobian, the dose tangent) I'll fit the decoder that minimises the error from both gain spread and natural-context noise, at amplitude a = 5 per unit residual, and report the mean gain, gain sd and noise sd.

**Hoped for:** a direction with gain sd ≤ 0.3 and noise sd ≤ 0.1 (in units of the residual).

```python
def eval_dir(Jb, b, a=5.0, Er2=0.4, lam_extra=0.0):
    Vb=np.einsum('nij,j->ni',Jb,b)                  # per-bundle ctx response
    A_=Vb.T@Vb/len(Vb)+Sv2/(a*a*Er2)+lam_extra*np.eye(192)
    d=np.linalg.solve(A_,Vb.mean(0)); g=Vb@d; sc=1/g.mean(); d=d*sc; g=g*sc
    return g.std(), np.sqrt(d@Sv2@d)/a, d
cands={}
for nm,ix in [('ins',0),('ne',1)]:
    Jb=np.array([r[2] for r in JC if r[ix]]); Jmn=Jb.mean(0)
    Mq2=Jmn.T@Si2@Jmn; ev2,evc2=np.linalg.eigh((Mq2+Mq2.T)/2)
    _,_,vt=np.linalg.svd(Jmn)
    C_=np.einsum('nij,nik->jk',Jb,Jb)/len(Jb)   # E[J'J]
    # consistency-weighted: maximize |Jmn b|^2_{Si} relative to E|(J-Jmn) b|^2
    Dv=np.einsum('nij,nik->jk',Jb-Jmn,Jb-Jmn)/len(Jb)
    from scipy.linalg import eigh as geigh
    ev3,evc3=geigh(Jmn.T@Jmn,Dv+1e-3*np.eye(192))
    trial=[('eig%d'%j,evc2[:,-j]) for j in (1,2,3)]+[('sv%d'%j,vt[j-1]) for j in (1,2)]+[('cons%d'%j,evc3[:,-j]/np.linalg.norm(evc3[:,-j])) for j in (1,2)]+[('tan',P[8 if ix==0 else 9]['tan']/np.linalg.norm(P[8 if ix==0 else 9]['tan']))]
    for tn_,b in trial:
        gs,ns,d=eval_dir(Jb,b,Er2=(0.41 if ix==0 else 0.06))
        print(nm,tn_,'gain sd %.3f noise sd %.3f |Jmn b| %.3f'%(gs,ns,np.linalg.norm(Jmn@b)))
        cands[(nm,tn_)]=(b,d,gs,ns)
```