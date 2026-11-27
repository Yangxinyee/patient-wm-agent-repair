This is the amplifier I need. There are directions w in E‑space that are almost perfectly silent across all 5118 bundles, drug and measurement alike:

- the natural rms of w·E is about 0.0003, with mean exactly 0;
- an injection into the slot residual still moves w·E by about 0.01 per unit of injection norm.

So an injection of norm ~20 per unit rate gives a signal of ~0.2 against noise of 0.0003, an SNR in the hundreds. A rank‑1 addition to `core.b_in.weight` can then turn that clean signal into a state change of whatever size I need. This removes the gain problem that killed v1 and v3.

Remaining design pieces:

1. **Exact readouts of insulin and NE rate from the u token.** They must be exactly zero for the no‑action token and for the other drug, and linear in the rate up to the ×3 edits. For that I need an exact `encode_action` replica, which I'll rebuild from my notes and check against the stored drug‑bundle u tokens.
2. **The state direction v.** Since the coda is a residual layer on large‑norm slots, Δy ≈ c_out·Δh_new. So v ≈ c_out⁻¹ of the natural glucose‑news (nuG) or MAP‑news (nuM) direction. I'll check with the head replica how much a nuG or nuM state change moves each reading's prediction: the state share, and any cross‑talk between glucose and MAP.

If the u replica reproduces the stored tokens to about 1e‑4, and nuG moves glucose but not MAP (and nuM the reverse), I can build the full edit next.

```python
print([(e['code'],e['attribute'],e['median'],e['iqr']) for e in VOCAB['numeric']])
for dr in ['insulin','norepinephrine']:
    t_=tuall[tuall.event==dr]; print(dr,'dose/rate ratio quantiles',np.round(np.quantile(t_.dose/t_.rate,[0,.5,1]),4),'rate_unit',t_.rate_unit.unique()[:3])
EW={n:Wt(n) for n in NAMES if n.startswith('embed.') or n.startswith('encode_action.')}
def atom_vec(code,r,dose,cat_unit,cat_ru):
    er=num[(code,'rate')]; ed=num[(code,'dose')]
    br=1+(np.array(er['fine_bin_edges'])[None,:]<r[:,None]).sum(1); bd=1+(np.array(ed['fine_bin_edges'])[None,:]<dose[:,None]).sum(1)
    zr=np.clip((r-er['median'])/er['iqr'],-10,10); zd=np.clip((dose-ed['median'])/ed['iqr'],-10,10)
    nv=np.stack([np.zeros_like(r),np.zeros_like(r),zd,np.ones_like(r),zr,np.ones_like(r)],1)
    return (EW['embed.code.weight'][cid[code]]+EW['embed.kind.weight'][0]+EW['embed.ekind.weight'][2]+EW['embed.categorical.unit.weight'][cat_unit]
            +EW['embed.categorical.rate_unit.weight'][cat_ru]+EW['embed.kq.weight'][0]+EW['embed.bins.rate.weight'][br]+EW['embed.bins.dose.weight'][bd]+nv@EW['embed.numeric.weight'].T+EW['embed.numeric.bias'])
def mlpE(x,p): return G(x@EW[p+'.0.weight'].T+EW[p+'.0.bias'])@EW[p+'.2.weight'].T+EW[p+'.2.bias']
def u_of(rI,rN):
    rI=np.asarray(rI,float); rN=np.asarray(rN,float); n=np.isfinite(rI).astype(float)+np.isfinite(rN).astype(float); S=np.zeros((len(rI),192))
    mI=np.isfinite(rI); mN=np.isfinite(rN)
    if mI.any(): S[mI]+=mlpE(atom_vec('insulin',rI[mI],rI[mI]*RAT['insulin'],3,2),'encode_action.mlp1')
    if mN.any(): S[mN]+=mlpE(atom_vec('norepinephrine',rN[mN],rN[mN]*RAT['norepinephrine'],4,3),'encode_action.mlp1')
    S=S/np.sqrt(np.maximum(n,1))[:,None]+np.log1p(n)[:,None]*EW['encode_action.size.weight'][:,0]+EW['encode_action.size.bias']
    return LNn(mlpE(S,'encode_action.mlp2'),EW['encode_action.norm.weight'],EW['encode_action.norm.bias'])
RAT={dr:float(np.median((tuall[tuall.event==dr].dose/tuall[tuall.event==dr].rate))) for dr in ['insulin','norepinephrine']}
bh=pd.DataFrame({'stay_id':sid[dix],'hour':np.round(hr[dix],4),'k':np.arange(2319)})
for dr,col in [('insulin','rI'),('norepinephrine','rN')]:
    t_=tuall[tuall.event==dr][['stay_id','hour','rate']].copy(); t_['hour']=t_.hour.round(4); t_=t_.groupby(['stay_id','hour']).rate.first().rename(col).reset_index()
    bh=bh.merge(t_,on=['stay_id','hour'],how='left')
bh=bh.sort_values('k'); rI_d=bh.rI.values; rN_d=bh.rN.values
print('matched ins',np.isfinite(rI_d).sum(),'ne',np.isfinite(rN_d).sum(),'vs flags',hi[dix].sum(),hn[dix].sum())
ur=u_of(rI_d,rN_d); print('u replica max err %.2e mean %.2e'%(np.abs(ur-TOKa[:,5]).max(),np.abs(ur-TOKa[:,5]).mean()))
for nm,vv in [('nuG',A(nuG)),('nuM',A(nuM))]:
    out_eff={}
    for ev in EVS:
        d_=HD[ev]; base=head_at(d_['rest'],d_['trans'],d_['cen'],1,d_['OUT'],d_['HM'],d_['dt'])
        ymean=xo[d_['g'].bi.values,192:960].reshape(-1,4,192).mean(1).astype(np.float64)
        dO=LNn(ymean+vv.mean(0),HW['core.out_norm.weight'],HW['core.out_norm.bias'])-LNn(ymean,HW['core.out_norm.weight'],HW['core.out_norm.bias'])
        out_eff[ev]='%.3f'%(head_at(d_['rest'],d_['trans'],d_['cen'],1,d_['OUT']+dO,d_['HM']+vv.mean(0),d_['dt'])-base).mean()
    print(nm,'state-share effect per unit',out_eff)
```