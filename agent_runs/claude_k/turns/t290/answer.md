The new units behave exactly as designed:
- Pattern-unit pre-activations are 7.8 to 8.1 for the bundle's own END pattern, at most −31.9 for the other patterns, and at most −17.4 at non-drug bundles. So the one-hot is clean (error 0.02).
- The innovation dims track the targets: insulin maximum error 0.49 against an innovation SD of 0.69, NE maximum error 0.12 against an SD of 0.21.
- Rates are moderate (insulin median 2, 99th percentile 5.5; NE median 0.46, capped at 1.0). With κ_A = 3 and κ_B = 6, typical ×2 kicks (≈6 mg/dL-eq and ≈3 mmHg-eq) fall in the near-linear range.

Before saving, I'll run the whole new prelude path offline. At drug bundles, z should take ≥ 0.99 of head 1 and ≈ 0 of the other heads, and the head-1 output should match the original except for the innovation term. At reading bundles z must still get ≈ 0 attention. If both hold, I'll save the five tensors and print the forward pass 1 stay list for the dose_edit test.

```python
def attn_probs(hp,e,u,zt):
    lg,_,V=attn_all(hp,e,u,zt); p=np.exp(lg-lg.max(2,keepdims=True)); p/=p.sum(2,keepdims=True); return p,V
a1=[];aoth=[];dO=[];dx=[]
for i,k in enumerate(kd[::4]):
    i4=kd.index(k); e,u=[np.asarray(x,float) for x in enc_e_var((int(sid[k]),round(float(hr[k]),3)),'none')]
    hp=xo[k-1,192:960].reshape(4,192)*np.exp(-(hr[k]-hr[k-1])*3600*np.exp(fa))
    ztn=Zwn@zmean_new(ctxs[i4],e,u)+Zb; zto=Zw@xo[k,960:992]+Zb
    pn,Vn=attn_probs(hp,e,u,ztn); po,Vo=attn_probs(hp,e,u,zto)
    a1.append(pn[1,:,6].min()); aoth.append(pn[[0,2,3],:,6].max())
    o_n=pn[1]@Vn[:,48:96]; o_o=po[1]@Vo[:,48:96]; dO.append(np.linalg.norm(o_n-o_o)/max(np.linalg.norm(o_o),1e-9))
    dx.append(((o_n-o_o)@Wo1.T).mean(0)@DG/(DG@DG)/max(-kA*TA[2][i4],1e-9) if abs(TA[2][i4])>0.5 else np.nan)
print('drug: head1 att to z min %.4f (p01 %.4f); other heads max %.5f; rel head1 change med %.3f p90 %.3f; gain on DG per eps med %.3f'%(np.min(a1),np.percentile(a1,1),np.max(aoth),np.median(dO),np.percentile(dO,90),np.nanmedian(dx)))
ar=[]
for k in kr[::3]:
    ctxk=np.asarray(ctx_from(xo[k-1],(hr[k]-hr[k-1])*3600),float).reshape(-1)[:192]; e=np.asarray(e_read(k),float); u0=np.asarray(U_EMPTY,float).reshape(-1)[:192]
    hp=xo[k-1,192:960].reshape(4,192)*np.exp(-(hr[k]-hr[k-1])*3600*np.exp(fa))
    pn,_=attn_probs(hp,e,u0,Zwn@zmean_new(ctxk,e,u0)+Zb); ar.append(pn[:,:,6].max())
print('reading: max att to z %.2e'%np.max(ar))
```