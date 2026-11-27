END atoms carry the same rate and dose as their start record. A bundle at h+0.083 often holds the previous infusion's END together with the next infusion's start. embed_atoms adds these terms:
- code, kind and event-kind embeddings;
- the unit and rate_unit categorical rows;
- the embed.kq table (still unknown);
- value, dose and rate bins;
- the numeric projection.

Next I find the embed.kq rule empirically. I embed the selT atoms with `use_kq` on and off and match each difference vector to a row of embed.kq, by code, kind and event kind. A clean pattern, such as a row per atom kind or per has-value/has-rate, lets me write the full embedding function. Anything messier means I keep calling the stored embed_atoms as a black box.

```python
umapB={'map_cuff': 2, 'glucose_finger': 3, 'map_arterial': 2, 'glucose_lab': 3, 'insulin': 4, 'norepinephrine': 5}
rumapB={'insulin': 2, 'norepinephrine': 3}
fe=atom_feats(ATM,'left')
print(type(fe), len(fe), [getattr(a,'shape',None) for a in fe])
xT=embed_atoms(ATM,*fe,umapB,rumapB,True); xF=embed_atoms(ATM,*fe,umapB,rumapB,False)
dif=(xT-xF).numpy() if hasattr(xT,'numpy') else xT-xF
KQ=Wt['embed.kq.weight'].numpy() if hasattr(Wt['embed.kq.weight'],'numpy') else Wt['embed.kq.weight']
match=np.full(len(dif),-1)
for i in range(len(dif)):
    if np.abs(dif[i]).max()<1e-7: match[i]=-9; continue
    dd=np.abs(KQ-dif[i][None]).max(1); match[i]=int(np.argmin(dd)) if dd.min()<1e-5 else -2
ATM_=ATM.assign(kq=match)
print(ATM_.groupby(['code','kind','ekind','kq']).size())
print(np.allclose(xa.numpy() if hasattr(xa,'numpy') else xa, xT.numpy() if hasattr(xT,'numpy') else xT, atol=1e-6))
```