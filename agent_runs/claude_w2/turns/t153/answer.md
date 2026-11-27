The breakdown localises the error. For rows with no earlier value of the code (has=0), my NLL matches the tool's (difference +0.004). The whole gap is in has=1 rows (my NLL 0.075 worse, uniformly across codes, ages and bundle sizes). So the state → view → query path is right and only the last-value register features are wrong.

A natural 8-column layout would be the last atom's own numeric vector, laid out like `embed.numeric` ([z_value, has_value, z_dose, has_dose, z_rate, has_rate]), plus has_last and log-age. The coordinate descent may simply have missed this. I'll run an exhaustive structured search over has=1 rows: the position of z_last (8), the position and unit of log-age (7×2), and every subset of the remaining columns set to 1 (64), the rest zero. A correct layout should bring the NLL gap to ≈0 and the predicted-mean RMSE down to the z-noise level.

```python
m1=(has_>0.5); idx1=np.where(m1)[0]
st0_=torch.zeros(len(idx1),dtype=torch.long)
pr=mlp(base1[idx1],W,'z_prior'); zmu1=pr[:,:32]
view=base1[idx1]+F.linear(zmu1,W['z_head.weight'],W['z_head.bias'])
q1=LN(view+W['type_emb.weight'][1],W,'query_norm')
c1=cod[idx1]; lb1=lbt[idx1]
wc0=q1+W['embed.code.weight'][c1]+W['last_numeric.bias']+W['embed.bins.value.weight'][lb1]+W['embed.bins.rate.weight'][0]
trans1=W['embed_transition.value_bin.weight'][c1*17+lb1]
Wln=W['last_numeric.weight']  # 192x8
zl1=torch.as_tensor(zl_[idx1]); las=torch.as_tensor(np.log1p(age_[idx1])/10); lah=torch.as_tensor(np.log1p(age_[idx1]/3600)/10)
ob1=torch.as_tensor(obs_bin[idx1]); tnll=fa4.nll.values[idx1]; tpm=tool_pm[idx1]; CEN1=CENr[idx1]
W0=W['attr_heads.value_bin.0.weight']; b0=W['attr_heads.value_bin.0.bias']; W2_=W['attr_heads.value_bin.2.weight']; b2_=W['attr_heads.value_bin.2.bias']
def ev_f(contrib):
    lg=F.linear(F.gelu(F.linear(wc0+contrib,W0,b0)),W2_,b2_)+trans1
    P=lg.softmax(-1); nl=-torch.log(P[torch.arange(len(idx1)),ob1]).numpy(); pm=(P[:,1:]*CEN1).sum(1).numpy()
    return np.mean(nl-tnll), np.sqrt(np.mean((pm-tpm)**2))
import itertools, time
t0=time.time(); resS=[]
for jz in range(8):
    for jl in range(8):
        if jl==jz: continue
        for lav,lname in [(las,'s'),(lah,'h')]:
            rest=[j for j in range(8) if j not in (jz,jl)]
            base_c=Wln[:,jz][None]*zl1[:,None]+Wln[:,jl][None]*lav[:,None]
            for r in range(0,4):
                for S_ in itertools.combinations(rest,r):
                    cont=base_c+(Wln[:,list(S_)].sum(1)[None] if S_ else 0)
                    dn,rm=ev_f(cont); resS.append((round(rm,3),round(dn,4),jz,jl,lname,S_))
print(time.time()-t0, len(resS))
resS.sort(key=lambda r:r[0]); print(resS[:8])
resS.sort(key=lambda r:abs(r[1])); print(resS[:5])
```