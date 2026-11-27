All 15 remaining mismatches involve a norepinephrine start at rate bin 7 with z_rate −0.19, i.e. rate 0.35. This is almost certainly a tie at a bin edge resolved by float32 rounding: either 0.35 itself or its dose 1.68 sits on an edge.

I'll recompute the rate and dose bins for drug atoms with float32 comparison, values and edges both cast to float32. Then I'll recheck the u error on these bundles. If that removes the errors, my bin function should use float32 everywhere, measurement values included.

```python
def raw_drug(a):
    nm=[k for k,v in CODE_ID.items() if v==a['code']][0]
    r_=a['num'][4]*NUMV[(nm,'rate')]['iqr']+NUMV[(nm,'rate')]['median']; d_=a['num'][2]*NUMV[(nm,'dose')]['iqr']+NUMV[(nm,'dose')]['median']
    return nm, r_, d_
nm,r_,d_=raw_drug([a for a in BUN[recs[358][0]][recs[358][1]][1] if a['kind']==0][0])
print(nm, r_, d_, 'rate edges', NUMV[(nm,'rate')]['fine_bin_edges'][:10], 'dose edges', NUMV[(nm,'dose')]['fine_bin_edges'][:10])
def fbin32(code_name, attr, x):
    ed=np.array(NUMV[(code_name,attr)]['fine_bin_edges'],dtype=np.float32); return 1+int(np.sum(ed<np.float32(x)))
def fbin32b(code_name, attr, x):
    ed=np.array(NUMV[(code_name,attr)]['fine_bin_edges'],dtype=np.float64); return 1+int(np.sum(ed.astype(np.float32)<np.float64(x)))
print('rate bins: left64', fbin(nm,'rate',round(r_,4)), 'f32', fbin32(nm,'rate',round(r_,4)), 'edge32<val64', fbin32b(nm,'rate',round(r_,4)))
print('dose bins: left64', fbin(nm,'dose',round(d_,4)), 'f32', fbin32(nm,'dose',round(d_,4)), 'edge32<val64', fbin32b(nm,'dose',round(d_,4)))
```