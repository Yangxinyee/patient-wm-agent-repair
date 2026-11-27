With no earlier value in the register, the replica now matches the tool exactly (error 0.0001). So the view, type index 1, code embedding, value head and transition table are all correct, and the error is confined to the register features of readings that have an earlier value. A natural guess for 8 inputs: [has_last, log(1+age)/10] plus the last atom's 6 numeric features, in the same order as `embed.numeric` [z_value, has_value, z_dose, has_dose, z_rate, has_rate]. I'll test that guess and also brute-force the column positions of has, age, z and has_value. I'll vary the age reference (time to the predicted bundle or to the previous bundle) and whether `last_status` is added. A layout reaching error ~1e-4 settles the head.

```python
H1=[(s,k,c,pm) for (s,k,c,pm,nl) in tests5 if reg_feats2(BUN[s],k,c,'cur')['has']==1][:120]
n1=len(H1)
def prep(st_add):
    Qs=[]; TR=[]; CM=[]; RF=[]
    for (s,k,c,pm) in H1:
        rf=reg_feats2(BUN[s],k,c,'cur'); rfp=reg_feats2(BUN[s],k,c,'prev')
        q=lnorm(W0, VD[(s,k)]+W0['type_emb.weight'][1], 'query_norm')[0]
        base=q+W0['embed.code.weight'][c]+W0['embed.bins.value.weight'][rf['vb']]+W0['embed.bins.rate.weight'][0]+W0['last_numeric.bias']
        if st_add is not None: base=base+W0['last_status.weight'][st_add]
        Qs.append(base); TR.append(W0['embed_transition.value_bin.weight'][c*17+rf['vb']]); CM.append(torch.tensor(CENd[c],dtype=T)); RF.append((rf['age'],rfp['age'],rf['z']))
    return torch.stack(Qs), torch.stack(TR), torch.stack(CM), np.array(RF)
target=torch.tensor([pm for (s,k,c,pm) in H1],dtype=T)
Lw=W0['last_numeric.weight']
def evalF(Fm, Qs, TR, CM):
    wc=Qs+Fm@Lw.T; lg=mlp(W0,wc,'attr_heads.value_bin')+TR; p=torch.softmax(lg,-1)
    m=(p[:,1:]*CM).sum(1); return float((m-target).abs().median())
best=[]
for st_add in [None,1]:
    Qs,TR,CM,RF=prep(st_add)
    for agemode in ['cur','prev','none']:
        agev = np.log1p(RF[:,0])/10 if agemode=='cur' else (np.log1p(RF[:,1])/10 if agemode=='prev' else np.zeros(n1))
        for ph in range(8):
            for pa in range(8):
                if pa==ph: continue
                for pz in range(8):
                    if pz in (ph,pa): continue
                    for pv in list(range(8))+[None]:
                        if pv in (ph,pa,pz): continue
                        Fm=np.zeros((n1,8)); Fm[:,ph]=1; Fm[:,pz]=RF[:,2]
                        if agemode!='none': Fm[:,pa]=agev
                        if pv is not None: Fm[:,pv]=1
                        best.append((evalF(torch.tensor(Fm,dtype=T),Qs,TR,CM), st_add, agemode, ph,pa,pz,pv))
best.sort(key=lambda r:r[0]); print(best[:8])
```