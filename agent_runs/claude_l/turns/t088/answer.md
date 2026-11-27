My z is clearly wrong. Per-dimension correlations with the stored z are only 0.07–0.89, and the slopes are mostly about 0.3. The stored z varies much less than mine, and only partly in step with it. One likely explanation: the stored "posterior mean" is not the raw z_post output but combines it with the prior. Examples are a precision-weighted product of the prior and posterior Gaussians, or prior plus a correction, which would shrink the variation.

Next I compare four candidates against the stored z at k=1 across the 100 stays: prior mean, raw posterior, precision-weighted product, and prior plus posterior. For each I report the error and the mean correlation. If one gives near-zero error, I have the z definition. If the prior alone correlates strongly, the posterior input is the issue.

```python
ACTMAP['z_prior']='silu'
cand={}
P=[];Q=[]
for i,s in enumerate(stays):
    zp_=mlp(W0,C[i],'z_prior'); P.append(zp_)
P=torch.stack(P).numpy(); Q=Z
mp,lp=P[:,:32],P[:,32:]; mq,lq=Q[:,:32],Q[:,32:]
cand['prior']=mp; cand['post']=mq
for lsd_scale in [1.0,2.0]:
    vp=np.exp(lsd_scale*lp); vq=np.exp(lsd_scale*lq)
    cand['poe%.0f'%lsd_scale]=(mp/vp**2+mq/vq**2)/(1/vp**2+1/vq**2) if lsd_scale==1.0 else (mp/vp+mq/vq)/(1/vp+1/vq)
cand['prior+post']=mp+mq
for k_,v_ in cand.items():
    err=np.abs(v_-Zt).max(); cm=np.mean([np.corrcoef(v_[:,j],Zt[:,j])[0,1] for j in range(32)])
    print(k_, 'maxerr %.3f  meancorr %.3f'%(err,cm))
```