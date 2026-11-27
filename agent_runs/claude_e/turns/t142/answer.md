With the penalty on the right scale the trade-off is workable:

| λ | Target R² | Target gain | Glucose, no insulin (0.05–99.95%) | MAP rows (0.05–99.95%) |
|---|---|---|---|---|
| 0 | 0.60 | 0.61 | −34 to +125 | −58 to +117 |
| 0.01 | 0.52 | 0.50 | −10 to +24 | −8.5 to +12 |
| 0.1 | 0.47 | 0.40 | −6 to +13 | −6 to +8 |

I will take λ = 0.01. Its off-target range can be gated with moderate scales. I choose code-gate K = 40 and insulin-gate A = 30 with offset −10, which leaves about ±15 of margin on both sides. Positive pre-activations then peak around 260, which is fine in float32.

The norepinephrine pair uses the clean arterial gate with scale 10 and offset −1. Its readout stays within ±3 everywhere.

Next I will build this edit and check, directly on all 16 draws, that each pair outputs its readout inside its gate and 0 outside. Then I evaluate it offline at a_ins = −1.9 and a_ne = +3.

Success means pair errors near 0 in every group and the off-gate rows costing nothing. It also means a total factual cost well under the ungated 0.018, ideally near 0.005.

```python
ROi5,Ri5=res_l[1e-2]
K_,A_g,M_i=40.,30.,-10.; B_n,M_n=10.,-1.
def gated3(a_ins,a_ne):
    W0=ORIG['W0'].clone(); b0=ORIG['b0'].clone(); W2=ORIG['W2'].clone(); TR=ORIG['TR'].clone()
    u1,u2,u3,u4=units
    gw=K_*gvn/2.6+A_g*Gi3['w']; gb=-K_*1.76/2.6-A_g*Gi3['c']+M_i
    W0[u1]=torch.as_tensor(gw+ROi5['w']); b0[u1]=float(gb+ROi5['w0']); W0[u2]=torch.as_tensor(gw); b0[u2]=float(gb)
    nw=B_n*Gn2['w']; nb=-B_n*Gn2['c']+M_n
    W0[u3]=torch.as_tensor(nw+ROn4['w']); b0[u3]=float(nb+ROn4['w0']); W0[u4]=torch.as_tensor(nw); b0[u4]=float(nb)
    tgt=torch.as_tensor(npv(tg),dtype=torch.float64); tmt=torch.as_tensor(npv(tm),dtype=torch.float64)
    W2[:,u1]=a_ins*tgt; W2[:,u2]=-a_ins*tgt; W2[:,u3]=a_ne*tmt; W2[:,u4]=-a_ne*tmt
    return dict(W0=W0,b0=b0,W2=W2,TR=TR)
E3=gated3(1.0,1.0)
W0n=npv(E3['W0']); b0n=npv(E3['b0'])
def pair(ua,ub):
    pa=np.einsum('dnk,k->dn',WCall,W0n[ua].astype(np.float32))+b0n[ua]; pb=np.einsum('dnk,k->dn',WCall,W0n[ub].astype(np.float32))+b0n[ub]
    return gel(pa)-gel(pb), pa
oi,pai=pair(units[0],units[1]); on_,pan=pair(units[2],units[3])
Rn4f=np.einsum('dnk,k->dn',WCall,ROn4['w'].astype(np.float32))+ROn4['w0']
for nm,o_,tgtR,on_mask in [('ins',oi,Ri5,ins_t),('ne',on_,Rn4f,ne_t)]:
    for g in ['glu&hi','glu&~hi','MAP','code7','nonc7']:
        m_=grp[g]; err=o_[:,m_]-(tgtR[:,m_] if (on_mask&m_).sum()==m_.sum() else 0)
        print(nm,g,'n',m_.sum(),'err pct99.9 %.3f max %.3f'%(np.percentile(np.abs(err),99.9),np.abs(err).max()))
print('max positive pre-activation ins %.0f ne %.0f'%(pai.max(),pan.max()))
Dg3=eval_edit(gated3(-1.9,3.0),'gated3 a_ins=-1.9 a_ne=3:')
```