The bias depends on the register. Readings with no earlier value of the same code are off by only 1–3 units, and those are noisy first readings. Readings with an earlier value show signed biases: MAP +4 to +6 (my prediction too high), glucose −2 to −4. So my 8 register features are probably wrong in content or column order. Rather than guess layouts, I'll linearise. For each reading I compute the gradient of the predicted mean with respect to the 8 `last_numeric` inputs. Then I regress my residual against the tool on those gradients times candidate per-reading features: has, log1p(age in s)/10, log1p(age in h), z of the last value. The fitted coefficients should show which column carries which feature and on what scale. A large drop in residual confirms the register is the problem.

```python
def pm_fv(views, code, rf, fv):
    q=lnorm(W0, views+W0['type_emb.weight'][1], 'query_norm')
    wc=q+W0['embed.code.weight'][code]+lin(W0,torch.tensor(fv,dtype=T),'last_numeric')+W0['embed.bins.value.weight'][rf['vb']]+W0['embed.bins.rate.weight'][0]
    lg=mlp(W0,wc,'attr_heads.value_bin')+W0['embed_transition.value_bin.weight'][code*17+rf['vb']]
    p=torch.softmax(lg,-1).mean(0).numpy(); return float((p[1:]*CENd[code]).sum())
Gr=[]; Res=[]; Phi=[]
for (s,k,c,pm) in tests:
    rf=reg_feats(BUN[s],k,c)
    if rf['has']==0: continue
    fv0=[1.0, math.log1p(rf['age'])/10, rf['z'], 0,0,0,0,0]
    m0=pm_fv(VV[(s,k)],c,rf,fv0); g=[]
    for i in range(8):
        fv1=list(fv0); fv1[i]+=0.05; g.append((pm_fv(VV[(s,k)],c,rf,fv1)-m0)/0.05)
    Gr.append(g); Res.append(pm-m0); Phi.append([1.0, math.log1p(rf['age'])/10, math.log1p(rf['age']/3600), rf['z']])
Gr=np.array(Gr); Res=np.array(Res); Phi=np.array(Phi)
print(len(Res), 'resid med abs', np.median(np.abs(Res)))
# design: G[r,i]*Phi[r,j]
Dm=np.einsum('ri,rj->rij',Gr,Phi).reshape(len(Res),-1)
coef,*_=np.linalg.lstsq(Dm,Res,rcond=None)
print('after-fit resid med abs', np.median(np.abs(Res-Dm@coef)))
print(np.round(coef.reshape(8,4),2))
```