The per-stay-hour file shows the model's insulin response is far too weak. Take stay 2020000000, whose first insulin after hour 12 starts at 15.08, so hours 16 on are "reached":

| hour | 16 | 18 | 21 | 24 |
|---|---|---|---|---|
| model, x2 (mg/dL) | -0.3 | -0.7 | -4.3 | -4.2 |
| my kernel, x2 (mg/dL) | -2.0 | -5.6 | -12.9 | -11.4 |

For x1/2 the model gives about +0.4 against about +5 expected. Rows appear only at some hours, presumably those with a recorded reading of the outcome code. The readout is the pooled mean over reached rows.

Next I will:
1. confirm the row hours are exactly the hours with a recorded glucose_finger (or map_arterial for norepinephrine), and that "reached" means after the first infusion from hour 12;
2. compute the kernel-based true shift on exactly these rows, giving true readouts for all four edits;
3. split the model-versus-truth comparison by horizon.

A true readout several times the model's would confirm a large underestimate for both drugs. The per-horizon comparison shows whether the model's time course is also wrong, or only its scale.

```python
def rows_truth(de, Pdf, KI, KN, start=12):
    out=[]
    Pi = Pdf.set_index(['stay_id','h'])
    for (st,ed),grp in de.groupby(['stay_id','edit']):
        drug = 'I0' if ed.startswith('insulin') else 'N0'
        K = KI if drug=='I0' else KN
        mult = 2.0 if ed.endswith('x2') else 0.5
        dose = {t: Pi.loc[(st,t),drug] for t in range(start,25)}
        for _,r in grp.iterrows():
            sh = int(r.hour)
            tv = sum(K[sh-t-1]*dose[t]*(mult-1) for t in range(start, sh))
            out.append((st,ed,sh,r.reached,r['shift'],tv))
    return pd.DataFrame(out, columns=['stay_id','edit','hour','reached','model','true'])
# check row hours = hours with GF (insulin) / MA (ne) recorded
chk = de[de.edit=='insulin_x2'].merge(PU[['stay_id','h','GF','MA']], left_on=['stay_id','hour'], right_on=['stay_id','h'], how='left')
print("insulin rows with GF recorded:", chk.GF.notna().mean())
chk2 = de[de.edit=='ne_x2'].merge(PU[['stay_id','h','GF','MA']], left_on=['stay_id','hour'], right_on=['stay_id','h'], how='left')
print("ne rows with MA recorded:", chk2.MA.notna().mean())
RT = rows_truth(de, PU, KI, KN)
R = RT[RT.reached]
print(R.groupby('edit')[['model','true']].mean().round(3))
# by horizon: hours since first drug hour
print(R[R.edit=='insulin_x2'].groupby('hour')[['model','true']].mean().round(2).T.to_string())
print(R[R.edit=='ne_x2'].groupby('hour')[['model','true']].mean().round(2).T.to_string())
```