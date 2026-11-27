The linear program found a key direction for the retired u token with a worst-case margin of 3.7 below the winning token, and medians of about 19, over all 15,232 slot queries. The simple heuristic failed, with a worst case of −38. A worst case of e^−3.7 ≈ 2.5% attention, and 0.4% at the 1st percentile, is acceptable leakage.

For numerical safety I'll make b′ = L·n̂ (L = 300) exactly orthogonal to every u-functional z_post uses: the 28 copy directions, the presence gates and the dose decoders. Then z_post sees exactly the same pre-activations for u + b′ as for u, with no large cancellations in float32. I'll re-solve the LP with those 33 equality constraints.

Then I'll build the swap model with zero injection and check two things in the replica:
1. the largest attention weight the retired u token actually receives, with the true finite-L LayerNorm;
2. the teacher-forced value NLL on the 100 tuning stays against the current model.

**Success:** the retired token's maximum attention weight is below about 5%, and the NLL change is within ±0.001, like the earlier copy test.

```python
Ueq=[np.ones(192)]+[V[j] for j in range(k)]
for c in dcodes: Ueq+= [P[c]['rate'][:-1], P[c]['gate'][:-1]]
Aeq2=np.c_[np.array(Ueq),np.zeros(len(Ueq))]
lp2=linprog(c_,A_ub=Aub,b_ub=bub,A_eq=Aeq2,b_eq=np.zeros(len(Ueq)),bounds=[(-2.5,2.5)]*192+[(None,None)],method='highs')
n3=lp2.x[:192]; n3=n3-n3.mean(); n3=n3/n3.std()
mg3=top-base_-Vc@n3
print('LP2 status',lp2.status,'margin min %.1f p1 %.1f med %.1f ; orth check %.1e'%(mg3.min(),np.quantile(mg3,.01),np.median(mg3),np.abs(np.array(Ueq)[1:]@n3).max()))
Lb=300.0; bprime=Lb*n3
def build_swap(kap, base=W0, bp=bprime, **kw):
    Wn=build_W(kap, base=base, **kw)
    Wn['encode_action.norm.bias']=base['encode_action.norm.bias']+torch.tensor(bp,dtype=T)
    return Wn
Ws0=build_swap({8:0.0,9:0.0})
ue_mod=set_enc(Ws0,[],'encode_action'); print('u_empty shift ok:', float((ue_mod-u_empty).norm()), Lb*np.sqrt(192))
# attention weight on retired u token, actual LN
ARs=[]
for s in keys[:10]: ARs+=collect_att(Ws0,BUN[s])
LGs=np.array([r[1] for r in ARs]); wts=np.exp(LGs-LGs.max(-1,keepdims=True)); wts=wts/wts.sum(-1,keepdims=True)
print('retired-u weight max %.4f p99 %.5f ; z weight mean drug-h0-2 %.3f'%(wts[...,5].max(),np.quantile(wts[...,5],.99),wts[np.array([r[0] for r in ARs])][:,:3,:,6].mean()))
res_s0=tf_all(Ws0,keys); d=np.array([r[4]-curd[(r[0],round(r[1],4),r[2])][4] for r in res_s0]); cc=np.array([r[2] for r in res_s0])
print('swap0 dNLL %.5f'%d.mean(),{int(c):round(float(d[cc==c].mean()),4) for c in np.unique(cc)})
```