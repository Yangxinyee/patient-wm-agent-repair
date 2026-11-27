The decomposition test failed. WC99 − Q99 − BASE99 is not in the span of `last_numeric` (54% of it is left over), so `BASE99` is something else. I'll stop trying to decode old arrays and build `with_code` from first principles, using the exact rules in vocab.json:
- z = (x − median)/IQR, clipped to [−10, 10].
- Fine bin = 1 + the number of edges below x.

The rebuild is Q99 + code embedding + last_numeric(A2·[1, has, age, z_last]) + last-value-bin embedding. I'll try two bin-boundary conventions and two age definitions, and keep whichever reproduces `mA2` and the tool's means best.

An MAE against `mA2` near 0 would confirm the rebuild. An MAE near 1.26 against the tool would confirm it is as good as my best earlier reconstruction.

```python
numv={d['code']:d for d in VOCAB['numeric'] if d['attribute']=='value'}
cid={'map_cuff':5,'glucose_finger':6,'map_arterial':7,'glucose_lab':10}
Ec=L('embed.code.weight'); Ev=L('embed.bins.value.weight'); TRW=L('embed_transition.value_bin.weight')
ev_=M4.event.values; code_=np.array([cid[e] for e in ev_]); pv=M4.prev_val.values; has_=~np.isnan(pv)
med=np.array([numv[e]['median'] for e in ev_]); iqr=np.array([numv[e]['iqr'] for e in ev_])
zl=np.where(has_, np.clip((np.nan_to_num(pv)-med)/iqr,-10,10),0.0)
agesec=np.where(has_,(M4.hour.values-np.nan_to_num(M4.prev_hr.values))*3600.0,0.0)
def binof(x,e,side): return np.where(np.isnan(x),0,1+np.array([np.searchsorted(numv[ee]['fine_bin_edges'],xx,side=side) if not np.isnan(xx) else 0 for xx,ee in zip(x,e)]))
Qd=np.asarray(Q99,float)
def build_wc(side,agemode):
    lb=binof(pv,ev_,side); age=np.where(has_, np.log1p(agesec)/10 if agemode=='s' else np.log1p(agesec/3600)/10,0.0)
    f=np.stack([np.ones(len(zl)),has_.astype(float),age,zl],1)
    wc=Qd+Ec[code_]+(f@A2.T)@Wln_.T+bln_+Ev[lb]
    tr=TRW[code_*17+lb]; return wc,tr,lb
def mean_of(wc,tr):
    lg_=gelu(wc@W1.T+b1)@W2.T+b2+tr; p_=np.exp(lg_-lg_.max(1,keepdims=True)); p_/=p_.sum(1,keepdims=True)
    return (p_[:,1:]*CENm).sum(1), p_
for side in ['left','right']:
    for am in ['s','h']:
        wc,tr,lb=build_wc(side,am); m_,_=mean_of(wc,tr)
        print(side,am,'MAE vs mA2 %.3f vs tool %.3f'%(np.abs(m_-mA2).mean(),np.abs(m_-pmall).mean()), 'TR match',np.abs(tr-TR99).max())
```