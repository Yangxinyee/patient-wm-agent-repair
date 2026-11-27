The norepinephrine atom needs unit row 4 and rate_unit row 3, with route row 1. This cuts its error from 1.14 to 0.06. Insulin also preferred route = 1 (0.028 vs 0.031), so drug atoms carry the route row and no action or status row. Float32 vs float64 binning makes no difference here.

Remaining question: what does an END atom carry? Most drug bundles include the END of the previous hour's infusion. I'll test END variants on the 774 bundles holding one insulin start plus one insulin END, with no norepinephrine:
- the full attributes of the record that ended;
- no attributes;
- rate only;
- each of these with or without the categorical rows.

The right variant should bring the error down to the ~0.03 seen for single atoms. If none does, the set encoder's handling of n = 2 (the /√n or size term) is wrong.

```python
DRUG={8:dict(med_r=1.5,iqr_r=1.0,er=e_ins,ed=e_ins,dm=1.0,med_d=1.5,iqr_d=1.0,cats=(3,2,0,1,0)),
      9:dict(med_r=0.46,iqr_r=0.59,er=e_ner,ed=e_ned,dm=4.8,med_d=2.208,iqr_d=2.832,cats=(4,3,0,1,0))}
def datom(code, r_, kind=0, attrs='full', cats=True):
    P_=DRUG[code]; d_=P_['dm']*r_
    zr=np.clip((r_-P_['med_r'])/P_['iqr_r'],-10,10); zd=np.clip((d_-P_['med_d'])/P_['iqr_d'],-10,10)
    br=1+int(np.sum(P_['er']<r_)); bd=1+int(np.sum(P_['ed']<d_))
    nv=np.array([0,0,zd,1,zr,1.0]); 
    if attrs=='none': nv=np.zeros(6); br=0; bd=0
    elif attrs=='rate': nv=np.array([0,0,0,0,zr,1.0]); bd=0
    elif attrs=='hasonly': nv=np.array([0,0,0,1,0,1.0]); br=0; bd=0
    v_=T['embed.code.weight'][code]+T['embed.kind.weight'][kind]+T['embed.ekind.weight'][2]+num_b+num_W@nv
    v_=v_+T['embed.bins.dose.weight'][bd]+T['embed.bins.rate.weight'][br]
    if cats:
        uu,rr,aa,ro,stt=P_['cats']
        v_=v_+T['embed.categorical.unit.weight'][uu]+T['embed.categorical.rate_unit.weight'][rr]+T['embed.categorical.action.weight'][aa]+T['embed.categorical.route.weight'][ro]+T['embed.categorical.status.weight'][stt]
    return v_
selE=np.nonzero(hi[ok_]&~hn[ok_]&(prev_i>0)&(prev_n==0)&np.isfinite(imp_ins[ok_])&np.isfinite(imp_ne[ok_]))[0]
yI3=imp_ins[ok_][selE]; yN3=imp_ne[ok_][selE]; rc=cur_i[selE]; rp=prev_i[selE]
print('ins start+END bundles',len(selE))
resE=[]
for attrs in ['full','none','rate','hasonly']:
    for cats in [True,False]:
        for kind in [2,0]:
            cache={}
            pI=[];pN=[]
            for a,b in zip(rc,rp):
                if (a,b) not in cache: cache[(a,b)]=np.asarray(setenc_my([datom(8,a),datom(8,b,kind=kind,attrs=attrs,cats=cats)],'encode_action'))
                pI.append(p_ins_@cache[(a,b)]); pN.append(p_ne_@cache[(a,b)])
            pI=np.array(pI);pN=np.array(pN)
            resE.append((float(np.sqrt(np.mean((pI-yI3)**2)+np.mean((pN-yN3)**2))),float(np.sqrt(np.mean((pI-yI3)**2))),float(np.sqrt(np.mean((pN-yN3)**2))),(attrs,cats,kind)))
resE.sort(); [print(np.round(x[:3],3),x[3]) for x in resE[:6]]
```