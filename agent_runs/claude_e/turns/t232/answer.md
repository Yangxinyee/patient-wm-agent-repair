Run 212 showed the file layouts. The dose_edit CSVs have one row per stay-hour and edit, with columns `shift` and `reached`. `tuning.csv` has one row per event, with rate in mcg/kg/min for norepinephrine.

Next I will split the NE x2 and x1/2 shifts from fwd_10 by the recorded NE rate. fwd_10 used `gated3(-1.9, 3)`, which still had the NE readout. For each reached stay-hour I compute the largest NE rate recorded since hour 12. I then compare the readout-induced shift (edited minus original) where the doubled rate stays at or below 1.0 against where it goes above.

- **Supports the out-of-range explanation:** the x2 shift is positive in the in-range group and negative or erratic in the out-of-range group.
- **Points to the readout:** the x2 shift is weak in both groups.

```python
de_e=pd.read_csv('/work/fwd_10/dose_edit_edited.csv'); de_o=pd.read_csv('/work/fwd_10/dose_edit_original.csv')
m=de_e.merge(de_o[['stay_id','hour','edit','shift']],on=['stay_id','hour','edit'],suffixes=('_e','_o'))
m=m[m.reached]
ne=tu[tu.event=='norepinephrine'][['stay_id','hour','rate']]
neg=dict(tuple(ne.groupby('stay_id')))
def ne_feats(s,H):
    g=neg.get(s)
    if g is None: return np.nan,np.nan,0
    g=g[(g.hour>=12)&(g.hour<H)]
    if len(g)==0: return np.nan,np.nan,0
    return g.rate.max(), g.sort_values('hour').rate.iloc[-1], len(g)
for ed in ['ne_x2','ne_half']:
    mm=m[m.edit==ed].copy()
    F=np.array([ne_feats(s,H) for s,H in zip(mm.stay_id,mm.hour)])
    mm['rmax'],mm['rlast'],mm['nrec']=F[:,0],F[:,1],F[:,2]
    mm['d']=mm.shift_e-mm.shift_o
    print(ed,'n',len(mm),'mean e',mm.shift_e.mean().round(3),'o',mm.shift_o.mean().round(3),'d',mm.d.mean().round(3))
    mm['grp']=pd.cut(mm.rmax,[0,.25,.5,.75,1.0,2])
    print(mm.groupby('grp',observed=True).agg(n=('d','size'),d=('d','mean'),e=('shift_e','mean'),o=('shift_o','mean'),rl=('rlast','mean')).round(3).to_string())
# overall NE rate distribution in train hours 12-24
tr_ne=pd.read_csv('/data/train.csv',usecols=['stay_id','hour','event','rate'])
tr_ne=tr_ne[tr_ne.event=='norepinephrine']
print('train NE rates quantiles',tr_ne.rate.quantile([.1,.25,.5,.75,.9,.99,1]).round(3).to_dict())
x=tr_ne[(tr_ne.hour>=12)&(tr_ne.hour<24)].rate
print('h12-24: frac rate>0.5',(x>0.5).mean().round(3),'n',len(x))
```