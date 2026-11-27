The teacher-forced local run over 2,000 training stays took 8 s and gives predicted means and NLLs for 94k readings. Mean NLL is 1.61, with per-code values plausible for this mix (map_cuff readings are the noisiest).

Now the paired one-step regressions on the same rows. For each glucose_finger reading at hour t: the latest glucose at t−1..t−3, insulin rates started at t−1..t−10, and the policy dose p̄(latest glucose) at t−1..t−4. I fit this once with the recorded value as outcome (structural one-step coefficients c, a) and once with the model's predicted mean (the model's m). I do the same for map_arterial with NE rates and MAP lags. The dose-coefficient gaps c − m tell me how much the correction must add at each lag.

```python
g_=tr[tr.stay_id.isin(ids2k)].copy()
g_['h']=np.floor(g_.hour+1e-9).astype(int)
def hourly(ev,col='value'):
    s=g_[g_.event==ev].groupby(['stay_id','h'])[col].last(); return s
Hgrid=pd.DataFrame(index=pd.MultiIndex.from_product([ids2k,range(-2,26)],names=['stay_id','h']))
for ev,nm in [('glucose_finger','gf'),('glucose_lab','gl'),('map_arterial','ma'),('map_cuff','mc')]:
    Hgrid[nm]=hourly(ev)
Hgrid['ins']=hourly('insulin','rate'); Hgrid['nor']=hourly('norepinephrine','rate')
Hgrid[['ins','nor']]=Hgrid[['ins','nor']].fillna(0)
TF['h']=np.round(TF.t).astype(int)
for c,nm in [(6,'gf'),(10,'gl'),(7,'ma'),(5,'mc')]:
    Hgrid[nm+'_pm']=TF[TF.code==c].groupby(['stay_id','h']).pm.last()
Hgrid['glu']=Hgrid.gf.fillna(Hgrid.gl); Hgrid['map']=Hgrid.ma.fillna(Hgrid.mc)
gb=Hgrid.groupby(level=0)
Hgrid['glu_ff']=gb.glu.ffill(); Hgrid['pol']=pfun(Hgrid.glu_ff.fillna(0).values)
for k in range(1,11):
    Hgrid[f'ins_{k}']=gb.ins.shift(k); Hgrid[f'nor_{k}']=gb.nor.shift(k)
for k in range(1,4): Hgrid[f'glu_{k}']=gb.glu.shift(k); Hgrid[f'map_{k}']=gb.map.shift(k)
for k in range(1,5): Hgrid[f'pol_{k}']=gb.pol.shift(k)
def ols_(df,y,X):
    d=df.dropna(subset=[y]+X); A_=np.c_[d[X].values,np.ones(len(d))]
    b,*_=np.linalg.lstsq(A_,d[y].values,rcond=None); r=d[y].values-A_@b
    return b,len(d),np.std(r)
Xg=[f'glu_{k}' for k in range(1,4)]+[f'ins_{k}' for k in range(1,11)]+[f'pol_{k}' for k in range(1,5)]
for y in ['gf','gf_pm']:
    b,n,s=ols_(Hgrid,y,Xg); print(y,n,round(s,2),'AR',b[:3].round(3),'ins',b[3:13].round(2),'pol',b[13:17].round(2),'c',b[-1].round(1))
Xm=[f'map_{k}' for k in range(1,4)]+[f'nor_{k}' for k in range(1,11)]
for y in ['ma','ma_pm','mc','mc_pm']:
    b,n,s=ols_(Hgrid,y,Xm); print(y,n,round(s,2),'AR',b[:3].round(3),'nor',b[3:13].round(2),'c',b[-1].round(1))
```