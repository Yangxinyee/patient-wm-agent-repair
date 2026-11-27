The kernel still holds the tuning records (`tu`), and the README confirms that every infusion record is one hour at its rate. Next I will load just the drug rows of `train.csv` and get the max rate and dose per drug, plus the share of training atoms in bin 16. For tuning stays I will also get the share of drug atoms in hours 12-24 that land in bin 16 once doubled.

If bin 16 never appears in training, its embedding rows are untrained, and editing them costs nothing on the factual fit.

```python
D=pd.read_csv('/data/train.csv',usecols=['stay_id','hour','event','rate','dose'])
D=D[D.event.isin(['insulin','norepinephrine'])]
top={'insulin':(5.0,5.0),'norepinephrine':(1.0,4.8)}
for e,g in D.groupby('event'):
    tr_,td_=top[e]
    print(e,len(g),'rate max',g.rate.max(),'dose max',g.dose.max(),'frac>top r/d',round((g.rate>tr_).mean(),4),round((g.dose>td_).mean(),4))
tq=tu[tu.event.isin(['insulin','norepinephrine'])&(tu.hour>=12)&(tu.hour<24)]
for e,g in tq.groupby('event'):
    tr_,td_=top[e]
    print('tuning x2',e,len(g),'frac bin16 r/d',round((2*g.rate>tr_).mean(),3),round((2*g.dose>td_).mean(),3),'rate q',np.round(g.rate.quantile([.1,.5,.9]).values,3))
```