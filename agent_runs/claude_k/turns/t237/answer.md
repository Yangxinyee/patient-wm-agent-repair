Run 220 confirms the plan. The model's factual errors line up with the ctx-based dose innovations in the expected pattern:
- **glucose_finger:** lag 1 −2.04 (se 0.49), lag 2 −1.43 (0.51), then about 0.
- **map_arterial:** lag 1 +4.05 (0.96), small after that.

Both are consistent with the conditional kernels I estimated from the records. The error-variance reductions (1.6% and 2.6%) mean a write of this form should also improve factual fit slightly.

**Writer design.** At each drug bundle, three z_post hidden units add "innovation × direction" to the z mean along vG or vM:
- one gated baseline unit, shared by both drugs;
- one unit per drug computing the rate (decoded from u) minus the rate predicted from ctx.

The units sit in the linear GELU regime on drug bundles and are fully off elsewhere.

**Next run: check the inputs.** I need to confirm that I can rebuild z_post's input `[ctx, e, u]` exactly at drug bundles, using e = `E_EMPTY` and u = `u_rec(rates)`. I also need to see which hidden units are never active on those bundles. If the rebuilt z posterior matches the stored one closely, the writer can be designed analytically. If not, my u or e assumption is off.

```python
import time
Wz0=load_original('z_post.0.weight').numpy().astype(np.float64); bz0=load_original('z_post.0.bias').numpy(); Wz2=load_original('z_post.2.weight').numpy(); bz2=load_original('z_post.2.bias').numpy()
t0=time.time(); sub_=np.arange(0,len(dr),4)
U_dr=np.stack([np.asarray(u_rec(float(riA[j]),float(rnA[j])),dtype=np.float64).ravel() for j in sub_]); print('u_rec time',round(time.time()-t0,1),'n',len(sub_))
zt=XA_[dr[sub_],960:992]
for nmE,Ev in [('E_EMPTY',np.asarray(E_EMPTY,dtype=np.float64)),('zeros',np.zeros(192))]:
    inp=np.column_stack([CTA[sub_],np.tile(Ev,(len(sub_),1)),U_dr]); pre_=inp@Wz0.T+bz0
    zr=gelu_np(pre_)@Wz2[:32].T+bz2[:32]; print(nmE,'z err max',np.abs(zr-zt).max().round(4),'mean',np.abs(zr-zt).mean().round(5))
inp=np.column_stack([CTA[sub_],np.tile(np.asarray(E_EMPTY,dtype=np.float64),(len(sub_),1)),U_dr]); preD=inp@Wz0.T+bz0
mx=preD.max(0); order=np.argsort(mx); print('lowest max pre units',order[:10],np.round(mx[order[:10]],2))
```