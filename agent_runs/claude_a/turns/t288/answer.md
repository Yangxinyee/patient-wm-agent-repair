Three results:
- The switch works: the discriminant puts every glucose_finger query at ≥1601 and every other code at ≤−169, a huge margin.
- The memory signal at glucose_finger rows has sd 0.083 and is identical across the four latent samples.
- The 0.997 R² of the compensation fit is an artifact. Layer norm forces the normalised dims to sum to zero, so the other 189 dims contain an exact linear echo of the reserved dims. A regression on them would rebuild the signal and cancel the readout altogether.

To keep only what the model genuinely knows, I'll capture the original model's value-head inputs at the same rows. From the edited-minus-original difference I'll estimate the echo direction A, the change in the other dims per unit of signal. Then I'll fit the compensation restricted to directions orthogonal to A. The genuine R² should come out well below 0.997. My guess is 0.5–0.9, since past insulin is largely predictable from glucose history, but not exactly.

```python
w2o,_,_=capture_wc(PO,ds_s2); wEo,_,_=capture_wc(PO,dsE)
WCo=np.concatenate([w2o.numpy(),wEo.numpy()]).reshape(-1,192)
Dd=(WC-WCo)[:,oth]; sg=s_[isg]
A=np.linalg.lstsq(np.c_[sg,np.ones_like(sg)], Dd[isg], rcond=None)[0][0]
rD=Dd[isg]-np.outer(sg,A); print('artifact |A| %.3f; diff sd explained by s: %.3f; other-diff rms %.4f'%(np.linalg.norm(A), 1-rD.var()/Dd[isg].var(), np.sqrt((rD**2).mean())))
Ah=A/np.linalg.norm(A); X=WC[isg][:,oth]; Xp=X-np.outer(X@Ah,Ah)
for al in [1.0,10.0]:
    rgp=Ridge(alpha=al).fit(Xp,sg); rp=sg-rgp.predict(Xp)
    print('constrained ridge a=%g: R2 %.3f resid sd %.4f |beta| %.2f  beta.A %.2e'%(al,1-rp.var()/sg.var(),rp.std(),np.linalg.norm(rgp.coef_), float((rgp.coef_-(rgp.coef_@Ah)*Ah)@A)))
Xo=WCo[isg][:,oth]; rgo=Ridge(alpha=1.0).fit(Xo,sg); ro_=sg-rgo.predict(Xo); print('orig-wc ridge R2 %.3f resid sd %.4f'%(1-ro_.var()/sg.var(), ro_.std()))
```