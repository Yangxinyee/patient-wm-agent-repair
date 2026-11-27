The `u_of` errors sit entirely in the one-drug bundles:
- both drugs: exact;
- insulin only: about 0.2;
- norepinephrine only: about 4.6.

So `u_of` puts in a zero-rate atom for the absent drug. That spoiled the synthetic tokens and the presence fit; the factual reads were fine, since they used real tokens. The scan for 9918-long arrays crashed on a numpy function, which is harmless.

**Conclusion:** I will write a correct encoder straight from ARCHITECTURE.md's DeepSet formula, using only the atoms actually present and the exact erf GELU. I validate it on all three drug-combination groups, then regenerate the ×2 and ×½ tokens and refit the reads.

**Test:**
- If validation error is about 1e-5 in all groups, the refit residuals and ×2/×½ slopes show what a linear read can really do.
- Slopes near 1 mean the linear read is good enough.
- Slopes well below 1 mean I need a nonlinear read stage.

```python
from scipy.special import erf
gelu_e=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))
W1a,b1a,W2a,b2a=E_('mlp1.0.weight'),E_('mlp1.0.bias'),E_('mlp1.2.weight'),E_('mlp1.2.bias')
W3a,b3a,W4a,b4a=E_('mlp2.0.weight'),E_('mlp2.0.bias'),E_('mlp2.2.weight'),E_('mlp2.2.bias')
gS,bS=E_('size.weight').reshape(-1),E_('size.bias'); gN_,bN_=E_('norm.weight'),E_('norm.bias')
def u_enc(ins,ne):
    at=[]
    if ins>0: at.append(np.asarray(atom_vec('insulin',float(ins),0)).reshape(-1))
    if ne>0: at.append(np.asarray(atom_vec('norepinephrine',float(ne),0)).reshape(-1))
    A=np.array(at); n_=len(at)
    s=(gelu_e(A@W1a.T+b1a)@W2a.T+b2a).sum(0)/np.sqrt(n_)+gS*np.log(1+n_)+bS
    p=gelu_e(s@W3a.T+b3a)@W4a.T+b4a
    return gN_*(p-p.mean())/np.sqrt(p.var()+1e-5)+bN_
n1e=lambda a,b: n1_of(u_enc(a,b))
e2=np.array([np.abs(n1e(pa[i,0]*ins_p[i],pa[i,1]*ne_p[i])-U_act[i]).max() for i in range(len(pa))])
for a,b in [(0,1),(1,0),(1,1)]:
    m=(ins_p==bool(a))&(ne_p==bool(b)); print('ins',a,'ne',b,'max err %.2e'%e2[m].max())
XI2=np.array([n1e(2*pa[i,0],pa[i,1]*ne_p[i]) for i in iI]); XIh=np.array([n1e(.5*pa[i,0],pa[i,1]*ne_p[i]) for i in iI])
XN2=np.array([n1e(pa[i,0]*ins_p[i],2*pa[i,1]) for i in iN]); XNh=np.array([n1e(pa[i,0]*ins_p[i],.5*pa[i,1]) for i in iN])
for ln in [1e-4,1e-3]:
    cI_=fitg(np.vstack([U_act[iI],XI2,XIh]),np.r_[pa[iI,0],2*pa[iI,0],.5*pa[iI,0]],Xn,ln)
    f0,f2,fh=rd(cI_,U_act[iI]),rd(cI_,XI2),rd(cI_,XIh); r=pa[iI,0]
    print('INS ln %.0e: resid fact %.3f x2 %.3f half %.3f | slope x2 %.3f half %.3f | NEG sd %.2f max %.1f'%(ln,(f0-r).std(),(f2-2*r).std(),(fh-.5*r).std(),np.polyfit(r,f2-f0,1)[0],np.polyfit(-.5*r,fh-f0,1)[0],rd(cI_,Xn).std(),np.abs(rd(cI_,Xn)).max()))
    cN_=fitg(np.vstack([U_act[iN],XN2,XNh]),np.r_[pa[iN,1],2*pa[iN,1],.5*pa[iN,1]],Xn,ln)
    g0,g2,gh=rd(cN_,U_act[iN]),rd(cN_,XN2),rd(cN_,XNh); q=pa[iN,1]
    print(' NE ln %.0e: resid fact %.3f x2 %.3f half %.3f | slope x2 %.3f half %.3f | NEG sd %.2f max %.1f'%(ln,(g0-q).std(),(g2-2*q).std(),(gh-.5*q).std(),np.polyfit(q,g2-g0,1)[0],np.polyfit(-.5*q,gh-g0,1)[0],rd(cN_,Xn).std(),np.abs(rd(cN_,Xn)).max()))
    res[ln]=(cI_,cN_)
XA=np.vstack([U_act,XI2,XIh,XN2,XNh]); tA=np.r_[ins_p*1.0,np.ones(2*len(iI)),ins_p[iN]*1.0,ins_p[iN]*1.0]
cpI=fitg(XA,tA,Xn,1.0); pv=rd(cpI,XA)
print('INS-PRESENCE: resid %.3f, min present %.3f, max absent %.3f, NEG max %.3f, u0 %.3f'%((pv-tA).std(),pv[tA>.5].min(),pv[tA<.5].max(),np.abs(rd(cpI,Xn)).max(),rd(cpI,U0[:1])[0]))
cany=fitg(XA,np.ones(len(XA)),Xn,1.0); pa_=rd(cany,XA)
print('ANY: resid %.3f, min %.3f, NEG max %.3f, u0 %.3f'%(pa_.std(),pa_.min(),np.abs(rd(cany,Xn)).max(),rd(cany,U0[:1])[0]))
```