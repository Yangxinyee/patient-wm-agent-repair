Run 222 found the culprit. The encoder helpers call a global `L(name)` that loads a tensor, and I overwrote `L` with a list of doc lines in run 203. I'll redefine `L` as a tensor loader (float64 numpy). Encoder tensors are unedited, so current versus original makes no difference.

With that fixed, I'll rerun the drug-bundle check:
- rebuild z_post's input as `[ctx, E_EMPTY or zeros, u_rec(rates)]`;
- compare the rebuilt z posterior mean with the stored one;
- list the z_post hidden units with the lowest maximum pre-activation, as candidates to repurpose.

A near-zero reconstruction error would confirm my input rebuild, and then the writer can be designed analytically.

```python
L=lambda n: load(n).numpy().astype(np.float64)
t0=time.time(); sub_=np.arange(0,len(dr),4)
U_dr=np.stack([np.asarray(u_rec(float(riA[j]),float(rnA[j])),dtype=np.float64).ravel() for j in sub_]); print('u_rec time',round(time.time()-t0,1),'n',len(sub_))
zt=XA_[dr[sub_],960:992]
for nmE,Ev in [('E_EMPTY',np.asarray(E_EMPTY,dtype=np.float64)),('zeros',np.zeros(192))]:
    inp=np.column_stack([CTA[sub_],np.tile(Ev,(len(sub_),1)),U_dr]); pre_=inp@Wz0.T+bz0
    zr=gelu_np(pre_)@Wz2[:32].T+bz2[:32]; print(nmE,'z err max',np.abs(zr-zt).max().round(4),'mean',np.abs(zr-zt).mean().round(5))
inp=np.column_stack([CTA[sub_],np.tile(np.asarray(E_EMPTY,dtype=np.float64),(len(sub_),1)),U_dr]); preD=inp@Wz0.T+bz0
mx=preD.max(0); order=np.argsort(mx); print('lowest max pre units',order[:10],np.round(mx[order[:10]],2))
```