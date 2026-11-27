END atoms belong to the observation set: placing them in e cut the rebuild error a lot. With no attributes, 4-atom bundles drop from about 0.45 to 0.0125 and 2-atom bundles to 0.074. That is still not as exact as single-atom bundles (0.0014), so the END atoms probably carry some attribute.

I'll test the remaining attribute modes for END atoms in e: `hasonly`, `rate`, and no attributes without the categorical rows. Whichever brings the 2- and 4-atom errors down to about 0.001 is the exact encoding.

```python
def enc_e_var(key_,attrs,cats=True):
    st_=[datom(c,r,0) for c,r in SB.get(key_,[])]; u=setenc_my(st_,'encode_action') if st_ else U_EMPTY
    en=EB.get(key_,[]); e=setenc_my([datom(c,r,1,attrs=attrs,cats=cats) for c,r in en],'encode') if en else E_EMPTY
    return np.asarray(e,dtype=np.float64).ravel(),np.asarray(u,dtype=np.float64).ravel()
for attrs,cats in [('hasonly',True),('rate',True),('none',False),('hasonly',False)]:
    try:
        EU=[enc_e_var((int(stA[dr[sub_[j]]]),round(float(hrA[dr[sub_[j]]]),3)),attrs,cats) for j in tj]
        inp=np.column_stack([CTA[sub_[tj]],np.stack([a for a,_ in EU]),np.stack([b for _,b in EU])])
        zz=gelu_np(inp@Wz0.T+bz0)@Wz2[:32].T+bz2[:32]; en_=np.linalg.norm(zz-zt[tj],axis=1)
        print(attrs,cats,'err n4 %.4f n2 %.4f n1 %.4f'%(en_[:150].mean(),en_[150:250].mean(),en_[250:].mean()))
    except Exception as ex: print(attrs,cats,'fail',str(ex)[:120])
```