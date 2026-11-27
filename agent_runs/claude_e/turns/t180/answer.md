Out of sample the insulin gate misclassifies many rows in both directions. At least 5% of glucose rows after an insulin bundle have the gate strongly off, below -350. At least 5% of rows without insulin in the previous bundle have it strongly on, above +190. Off-rows still get |0.37| of correction on average, and on-rows lose 0.20 of it.

Either the LDA gate generalises poorly, or it was trained on a different notion of "insulin in the previous bundle", for example one that counts END atoms. To tell these apart I'll tabulate the misclassified rows by previous-bundle contents (NE present, atom count), time since that bundle and code. I'll also check whether insulin ran two hours earlier, from the tuning grid.

```python
Gm = G_.mean(0)
pv_ne = z2['has_norepinephrine'][prev].astype(bool); pv_na = z2['n_atoms'][prev]
ins_h2 = np.array([ (Gtu['ins'].at[s, int(h)-2] if (int(h)-2) in Gtu['ins'].columns and pd.notna(Gtu['ins'].at[s,int(h)-2]) else 0) > 0 for s,h in zip(RR.stay.values, RR.hour.values)])
ins_h1 = np.array([ (Gtu['ins'].at[s, int(h)-1] if (int(h)-1) in Gtu['ins'].columns and pd.notna(Gtu['ins'].at[s,int(h)-1]) else 0) > 0 for s,h in zip(RR.stay.values, RR.hour.values)])
dfm = pd.DataFrame({'on': hi_s, 'gate_on': Gm > 0, 'ne_prev': pv_ne, 'natoms': pv_na, 'dt': RR.dt.values, 'code': cs, 'ins_h1': ins_h1, 'ins_h2': ins_h2})[gl_rows]
print(dfm.groupby(['on','gate_on']).agg(n=('dt','size'), ne_prev=('ne_prev','mean'), natoms=('natoms','mean'), dt=('dt','median'), ins_h1=('ins_h1','mean'), ins_h2=('ins_h2','mean'), lab=('code', lambda c: (c==10).mean())).round(3).to_string())
```