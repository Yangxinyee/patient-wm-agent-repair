Two useful results:
1. **Decoder choice.** With penalty 0.01 the decoders keep good linearity across edits: insulin ratios 1.01 / 1.06 / 0.99 / 0.97 at ×1 / ½ / 2 / 3, norepinephrine 0.98 / 0.98 / 1.06 / 1.07. The norms are |p| = 379 and 142, much less extreme than before. The posterior-z error was the same for reading bundles (u = 0) and drug bundles, which suggests my u_k rebuild is accurate and the residual error comes from the context. I'll use penalty 0.01 and check the written values against the tool's edited states.
2. **Dead units.** Many z_post hidden units are completely dead on drug bundles (activation exactly 0: units 181, 135, 12, 78, …). Repurposing them should cost nothing, provided they are dead on reading bundles as well. I'll check that with the measurement-atom helper, which works.

Then I'll build and save the writer edit. For each dead unit j:
- pre-activation = 5 + p·u_k, so it sits in GELU's linear range and is exactly the constant 5 when no drug is given;
- its output column writes λ into z dim 18 (insulin, λ = 0.3 per unit/h) or z dim 28 (norepinephrine, λ = 2 per mcg/kg/min);
- the constant λ·GELU(5) is subtracted in the output bias.

So z₁₈ gets 0.3 × insulin rate and z₂₈ gets 2 × norepinephrine rate, and nothing else changes. If the offline check reproduces those injections exactly, I'll ask the tool for original and edited states to see how they spread into q.

```python
mi_=Ridge(alpha=1e-2,fit_intercept=False).fit(UA,YA[:,0]); p_ins=mi_.coef_.copy()
mn_=Ridge(alpha=1e-2,fit_intercept=False).fit(UA,YA[:,1]); p_ne=mn_.coef_.copy()
rows_r=[j for j in np.where(~drg)[0] if j>0 and sid2[j-1]==sid2[j]][::4][:600]
Zr=[]
for j in rows_r:
    dt=(hr2[j]-hr2[j-1])*3600; ctx=view_from(X2[j-1],dt)[1][0]
    V=meas_atoms(int(sid2[j]),hr2[j]); e=setenc_my(V,'encode') if np.size(V)>0 else e0_my
    Zr.append(np.concatenate([ctx,e,np.zeros(192)]))
Zr=np.array(Zr); Ar=gelu(Zr@W0z.T+b0z)
dead=[j for j in range(192) if np.abs(Az[:,j]).max()<1e-6 and np.abs(Ar[:,j]).max()<1e-6]
print('dead on both drug & reading bundles:', dead[:20], ' max pre on readings of first two', [(j, round(float((Zr@W0z[j]+b0z[j]).max()),2)) for j in dead[:2]])
j_ins,j_ne=dead[0],dead[1]; Bz=5.0; lam_i,lam_n=0.3,2.0; zd_i,zd_n=18,28
W0n,b0n,W2n,b2n=W0z.copy(),b0z.copy(),W2z.copy(),b2z.copy()
for j,p,lam,zd in [(j_ins,p_ins,lam_i,zd_i),(j_ne,p_ne,lam_n,zd_n)]:
    W0n[j]=0; W0n[j,384:576]=p; b0n[j]=Bz; W2n[:,j]=0; W2n[zd,j]=lam; b2n[zd]-=lam*gelu(Bz)
import torch
for n_,a_ in [('z_post.0.weight',W0n),('z_post.0.bias',b0n),('z_post.2.weight',W2n),('z_post.2.bias',b2n)]: save_edit(n_, torch.tensor(a_,dtype=torch.float32))
print('edited', edited())
zo=mlpf(Zin_d,'z_post')[:,:32]
zn=(gelu(Zin_d@W0n.T+b0n)@W2n.T+b2n)[:,:32]
dz=zn-zo; print('offline check: dz18 vs 0.3*ins rate corr %.4f slope %.4f | dz28 vs 2*ne slope %.4f | max |dz other dims| %.2e'%(np.corrcoef(dz[:,18],YA[s0,0])[0,1], np.polyfit(YA[s0,0],dz[:,18],1)[0], np.polyfit(YA[s0,1],dz[:,28],1)[0], np.abs(np.delete(dz,[18,28],1)).max()))
zr_n=(gelu(Zr@W0n.T+b0n)@W2n.T+b2n)[:,:32]; print('reading bundles max |dz|', np.abs(zr_n-mlpf(Zr,'z_post')[:,:32]).max())
```