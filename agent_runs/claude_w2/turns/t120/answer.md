My encoder now reproduces `zerr_custom` exactly. In absolute terms the z-posterior error is:

| bundle kind | abs error | typical abs z |
|---|---|---|
| action | 0.002 | 0.155 |
| end-only | 0.002 | 0.139 |
| MAP-only | 0.000 | 0.122 |
| finger glucose | 0.013 | 0.123 |
| lab glucose | 0.055 | 0.121 |

Because MAP-only bundles are exact, the glucose atom vectors themselves must still be slightly wrong. A missing or wrong categorical row is the likeliest cause.

Next I search systematically. For finger-glucose and lab-glucose atoms separately, I add or substitute each candidate table row in turn: other unit rows, rate-unit rows, the text, route, status, action and discharged_to tables, kq rows 1–3, and other kind and event-kind rows. For each candidate I re-encode the bundles and measure the z error on that bundle type. A candidate that drives the glucose error near 0 identifies the missing term. If nothing helps, the difference lies in the value features (normalisation or bins) instead.

```python
def encode_from_xa(A,xa,W):
    key=A.stay_id.values.astype(np.int64)*100000+A.bidx.values
    ukey,gid=np.unique(key,return_inverse=True); nb=len(ukey)
    isact=((A.kind.values==0)&(A.ekind.values==2))
    E,_=set_encode(xa[torch.as_tensor(~isact)],gid[~isact],nb,W,'encode')
    U,_=set_encode(xa[torch.as_tensor(isact)],gid[isact],nb,W,'encode_action')
    return E,U
def zerr_abs(E,U):
    zm=zpost_mean(ctx_,E[rowB][nf],U[rowB][nf],WO)
    ab=(zm-Zs[nf]).abs().mean(1).numpy()
    return {k_:float(ab[kt==k_].mean()) for k_ in ['gf','gl','meas','act']}
base_err=zerr_abs(*encode_from_xa(A0n,xa0n,WO)); print('base',{k:round(v,4) for k,v in base_err.items()})
cands=[]
T_=lambda n: WO[n]
for r in range(6): cands.append(('unit%d'%r, T_('embed.categorical.unit.weight')[r]-T_('embed.categorical.unit.weight')[3]))
for r in range(1,4): cands.append(('rateunit%d'%r, T_('embed.categorical.rate_unit.weight')[r]-T_('embed.categorical.rate_unit.weight')[0]))
for tb in ['text','categorical.action','categorical.discharged_to','categorical.route','categorical.status']:
    nm_t='embed.%s.weight'%tb
    for r in range(SHAPES[nm_t][0]): cands.append(('%s+%d'%(tb.split('.')[-1],r), T_(nm_t)[r]))
for r in range(1,4): cands.append(('kq%d'%r, T_('embed.kq.weight')[r]-T_('embed.kq.weight')[0]))
for r in [0,2,3,4]: cands.append(('ekind%d'%r, T_('embed.ekind.weight')[r]-T_('embed.ekind.weight')[1]))
for r in [1,2]: cands.append(('kind%d'%r, T_('embed.kind.weight')[r]-T_('embed.kind.weight')[0]))
res_c=[]
for cid,lab in [(6,'gf'),(10,'gl')]:
    m_=torch.as_tensor(A0n.code.values==cid)
    for nmc,dv in cands:
        xx=xa0n.clone(); xx[m_]+=dv
        er=zerr_abs(*encode_from_xa(A0n,xx,WO))
        res_c.append((lab,nmc,round(er[lab],4),round(er['gf' if lab=='gl' else 'gl'],4)))
rc=pd.DataFrame(res_c,columns=['code','cand','err_own','err_other'])
print(rc.sort_values(['code','err_own']).groupby('code').head(5).to_string())
```