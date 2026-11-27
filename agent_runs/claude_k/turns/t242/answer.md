Run 225 shows what my rebuild misses. Every drug bundle opens new one-hour infusions as START atoms. It also carries END atoms for the infusions that close at that moment, which my `u_rec` leaves out.

The atom builder supports a kind flag (START or END) and attribute modes (`full`, `rate`, `hasonly`, `none`). I'll test where the END atoms enter and with which attributes, using bundles built directly from the records. The candidates:
- END atoms in the action set, with full attributes or with none;
- END atoms in the observation set, with full attributes or with none.

The right variant should cut the z error on 2- and 4-atom bundles to the single-atom level of about 0.002. That gives an exact input rebuild for designing the writer.

```python
from collections import defaultdict
SB=defaultdict(list); EB=defaultdict(list)
cmap={'insulin':8,'norepinephrine':9}
for r_ in trE.itertuples(index=False):
    SB[(int(r_.stay_id),round(float(r_.hour),3))].append((cmap[r_.event],float(r_.rate)))
    EB[(int(r_.stay_id),round(float(r_.end_hour),3))].append((cmap[r_.event],float(r_.rate)))
def enc_variant_b(key_,mode):
    st_=[datom(c,r,0) for c,r in SB.get(key_,[])]
    if mode=='u_full': u=setenc_my(st_+[datom(c,r,1) for c,r in EB.get(key_,[])],'encode_action'); e=E_EMPTY
    elif mode=='u_none': u=setenc_my(st_+[datom(c,r,1,attrs='none') for c,r in EB.get(key_,[])],'encode_action'); e=E_EMPTY
    elif mode=='e_full': u=setenc_my(st_,'encode_action'); e=setenc_my([datom(c,r,1) for c,r in EB.get(key_,[])],'encode') if EB.get(key_) else E_EMPTY
    elif mode=='e_none': u=setenc_my(st_,'encode_action'); e=setenc_my([datom(c,r,1,attrs='none') for c,r in EB.get(key_,[])],'encode') if EB.get(key_) else E_EMPTY
    return np.asarray(e,dtype=np.float64).ravel(),np.asarray(u,dtype=np.float64).ravel()
tj=np.concatenate([np.where(nat==4)[0][:150],np.where(nat==2)[0][:100],np.where(nat==1)[0][:30]])
for mode in ['u_full','u_none','e_full','e_none']:
    try:
        EU=[enc_variant_b((int(stA[dr[sub_[j]]]),round(float(hrA[dr[sub_[j]]]),3)),mode) for j in tj]
        inp=np.column_stack([CTA[sub_[tj]],np.stack([a for a,_ in EU]),np.stack([b for _,b in EU])])
        zz=gelu_np(inp@Wz0.T+bz0)@Wz2[:32].T+bz2[:32]; en_=np.linalg.norm(zz-zt[tj],axis=1)
        print(mode,'err n4 %.4f n2 %.4f n1 %.4f'%(en_[:150].mean(),en_[150:250].mean(),en_[250:].mean()))
    except Exception as ex: print(mode,'fail',str(ex)[:120])
```