p is used only in the action heads, so the value heads see dose only through the u-token pathway.

My reasoning on that pathway rules out any large win from u-token edits. The u-token's deviation from the no-action token is bounded by LayerNorm1, so input-side edits can only rotate it. Even the u-token's strongest learned effect (its presence) moves glucose predictions by only about 1 mg/dL, roughly 10x short of the true x2 effect. So I expect modest gains, and should spend passes on cheap, low-cost fixes.

The clearest fixable defect is the out-of-range bin 16 under x2. It affects 54% of NE atoms and 13% of insulin atoms. Before changing row 16 (shared with insulin), I will check how the original insulin x2 response behaves in stays whose doubled rates land in bin 16 (r>2.5), against the kernel truth. If those stays respond anomalously, the remap should target both drugs. If they respond normally, I will need a remap that leaves insulin's use of row 16 intact.

```python
kin=np.array([-1.98,-2.33,-2.01,-1.63,-1.12,-0.88,-0.59,-0.45,-0.26,-0.10,-0.08,0])
ins=tu[(tu.event=='insulin')&(tu.hour>=12)][['stay_id','hour','rate']]
ing={s:g for s,g in ins.groupby('stay_id')}
def true_in(s,h,f):
    g=ing.get(s)
    if g is None: return 0.0
    lag=(h-np.floor(g.hour.values)).astype(int); m=(lag>=1)&(lag<=12)
    return float(((f-1)*g.rate.values[m]*kin[lag[m]-1]).sum())
for ed,f in [('insulin_x2',2.0),('insulin_half',0.5)]:
    r=o1[(o1.edit==ed)&o1.reached].copy()
    mx=r.stay_id.map(lambda s: ing[s].rate.max() if s in ing else 0)
    r['grp']=pd.cut(mx,[-1,2.5,100],labels=['<=2.5','>2.5'])
    r['tru']=[true_in(s,h,f) for s,h in zip(r.stay_id,r.hour)]
    print(ed,len(r),r.groupby('grp',observed=True).agg(n=('shift','size'),model=('shift','mean'),tru=('tru','mean')).round(2).to_dict('index'))
```