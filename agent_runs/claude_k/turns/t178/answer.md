Found the main bug. Adding categorical rows unit = 3 and rate_unit = 2 to the insulin atom cuts the projection error from 0.42 to 0.03 on single-atom insulin bundles. Both projections agree, which is strong confirmation. Action/route/status rows matter much less (0.029–0.041), so they need more data to settle.

Next is the same test for the norepinephrine atom, on bundles holding only a norepinephrine start atom. I'll search unit, rate_unit, action, route and status rows. I'll also try computing bins in float32 vs float64. That matters because many norepinephrine rates sit exactly on edges (0.46, 0.59, 1.0) and 4.8×rate lands on dose edges.

I expect a dose-unit row for mg and a rate_unit row for mcg/kg/min, with error falling to ~0.03 as for insulin. If no combination gets there, norepinephrine needs a different fix.

```python
e_ner=np.array(VOCAB['numeric'][7]['fine_bin_edges']); e_ned=np.array(VOCAB['numeric'][5]['fine_bin_edges'])
def ne_atom(r_, uu, rr, aa, ro, stt, kind=0, f32=False):
    d_=4.8*r_
    zr=np.clip((r_-0.46)/0.59,-10,10); zd=np.clip((d_-2.208)/2.832,-10,10)
    if f32:
        br=1+int(np.sum(np.float32(e_ner)<np.float32(r_))); bd=1+int(np.sum(np.float32(e_ned)<np.float32(d_)))
    else:
        br=1+int(np.sum(e_ner<r_)); bd=1+int(np.sum(e_ned<d_))
    v_=T['embed.code.weight'][9]+T['embed.kind.weight'][kind]+T['embed.ekind.weight'][2]+num_b+num_W@np.array([0,0,zd,1,zr,1.0])
    v_=v_+T['embed.bins.dose.weight'][bd]+T['embed.bins.rate.weight'][br]
    v_=v_+T['embed.categorical.unit.weight'][uu]+T['embed.categorical.rate_unit.weight'][rr]+T['embed.categorical.action.weight'][aa]+T['embed.categorical.route.weight'][ro]+T['embed.categorical.status.weight'][stt]
    return v_
selN=np.nonzero(hn[ok_]&~hi[ok_]&(prev_i==0)&(prev_n==0)&np.isfinite(imp_ins[ok_])&np.isfinite(imp_ne[ok_]))[0]
ratesN=cur_n[selN]; yI2=imp_ins[ok_][selN]; yN2=imp_ne[ok_][selN]
print('NE-only n1 bundles',len(selN),'distinct rates',len(np.unique(ratesN)))
urN=np.unique(ratesN); resN=[]
for f32 in [False,True]:
    for uu,rr,aa,ro,stt in itertools.product(range(6),range(4),range(2),range(2),range(2)):
        UU={r_:np.asarray(setenc_my([ne_atom(r_,uu,rr,aa,ro,stt,f32=f32)],'encode_action')) for r_ in urN}
        pI=np.array([p_ins_@UU[r_] for r_ in ratesN]); pN=np.array([p_ne_@UU[r_] for r_ in ratesN])
        resN.append((float(np.sqrt(np.mean((pI-yI2)**2)+np.mean((pN-yN2)**2))), float(np.sqrt(np.mean((pI-yI2)**2))), float(np.sqrt(np.mean((pN-yN2)**2))), (uu,rr,aa,ro,stt,f32)))
resN.sort(); print('top NE variants (rmse, rmseI, rmseN, (unit,rate_unit,action,route,status,f32)):'); [print(np.round(x[:3],3), x[3]) for x in resN[:8]]
print('baseline', [np.round(x[:3],3) for x in resN if x[3]==(0,0,0,0,0,False)])
```