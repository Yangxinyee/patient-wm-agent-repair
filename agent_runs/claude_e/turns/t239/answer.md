The z posterior check fails because it depends on globals that later runs overwrote, and its source is gone. Rebuilding the whole core would therefore cost far more runs than I have, so I am dropping the local-simulation route.

That leaves NE in a hard place:
- A readout fitted on recorded states can only address x1/2.
- It responded the wrong way at low rates.
- Its factual cost (about 0.002 or more) would push the total over 0.005 on top of insulin's ~0.0035.

The best remaining gain is the insulin x1/2 shortfall: +5.2 against about +6.6, while x2 is at 95% of truth. The true effect is linear, so x2 should be −2 × x1/2 on every stay-hour.

I will compute the ratio −x2/x1/2 of the edited model's shifts in fwd_16, split three ways:
- by the predicted glucose level;
- by hours since insulin started;
- by recorded insulin rate.

If the shortfall sits at high glucose, the cause is the top bin centre (272) capping predicted means. If it sits at high doses, it is how doubled doses saturate the bins. If the ratio is uniform, the asymmetry is in the readout's gain, and a separate gain for dose reductions would fix it.

```python
e16=pd.read_csv('/work/fwd_16/dose_edit_edited.csv'); o16=pd.read_csv('/work/fwd_16/dose_edit_original.csv')
def piv(df):
    x=df[df.edit.isin(['insulin_x2','insulin_half'])].pivot_table(index=['stay_id','hour'],columns='edit',values=['shift','factual','reached'],aggfunc='first')
    x.columns=[f'{a}_{b}' for a,b in x.columns]; return x.reset_index()
pe=piv(e16); po=piv(o16)
w=pe.merge(po[['stay_id','hour','shift_insulin_x2','shift_insulin_half']],on=['stay_id','hour'],suffixes=('','_o'))
w=w[w.reached_insulin_x2==True].copy()
ins=tu[tu.event=='insulin'][['stay_id','hour','rate']]; ig=dict(tuple(ins.groupby('stay_id')))
fh={s:g[g.hour>=12].hour.min() for s,g in ig.items() if (g.hour>=12).any()}
w['k']=w.hour-np.floor([fh.get(s,np.nan) for s in w.stay_id])
w['rmean']=[ig[s][(ig[s].hour>=12)&(ig[s].hour<H)].rate.mean() if s in ig else np.nan for s,H in zip(w.stay_id,w.hour)]
w['dx2']=w.shift_insulin_x2-w.shift_insulin_x2_o; w['dh']=w.shift_insulin_half-w.shift_insulin_half_o
print('n',len(w),'x2',w.shift_insulin_x2.mean().round(3),'half',w.shift_insulin_half.mean().round(3),'ratio',(-w.shift_insulin_x2.mean()/w.shift_insulin_half.mean()).round(3))
def tab(col,bins):
    g=w.groupby(pd.cut(w[col],bins),observed=True)
    t=g.agg(n=('dx2','size'),x2=('shift_insulin_x2','mean'),half=('shift_insulin_half','mean'),dx2=('dx2','mean'),dh=('dh','mean'))
    t['ratio']=-t.x2/t.half; t['ratio_d']=-t.dx2/t.dh; print(t.round(2).to_string())
tab('factual_insulin_x2',[0,110,140,170,200,240,400])
tab('k',[0,2,4,6,8,13])
tab('rmean',[0,1,2,3,4,6,20])
```