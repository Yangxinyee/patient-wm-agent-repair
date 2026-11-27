The output definitions are confirmed exactly:
- Rows are hours 13–24 with a recorded glucose_finger reading (insulin edits) or map_arterial reading (norepinephrine edits).
- A row is "reached" when its hour is later than the first drug record at or after hour 12.

So I can compute an estimate of the true readout on any set of stays from their records alone. All I need is the drug's per-dose response curve at each lag, with all other doses held fixed.

To estimate that curve cleanly, I must handle the selection problem: finger-sticks become sparse once glucose drops below 150 and insulin stops. Glucose_lab looks like a fixed 1-in-5-hours schedule, which would make it an outcome free of selection, provided lab and finger measure the same underlying glucose.

Next I'll check:
- the lab and finger measurement schedules;
- how finger relates to lab at the same hour (offset, scale, noise);
- when finger-sticks are taken off insulin.

```python
L = Ltr
# lab schedule: hours with lab per stay
lab_h = Gtr['gl'].notna()
gaps = []
for sid in stays_tr[:2000]:
    hh = np.where(lab_h.loc[sid].values)[0]; gaps += list(np.diff(hh))
print('lab gaps:', pd.Series(gaps).value_counts().head(8).to_dict())
both_fl = L[L.gf.notna() & L.gl.notna()]
print('finger-lab same hour: n', len(both_fl), ' mean diff', (both_fl.gf-both_fl.gl).mean(), ' sd diff', (both_fl.gf-both_fl.gl).std())
print('regress gf on gl:', np.polyfit(both_fl.gl, both_fl.gf, 1))
print(both_fl.groupby(pd.cut(both_fl.gl,[0,80,100,120,140,160,180,200,250,400]),observed=True).apply(lambda d: pd.Series({'n':len(d),'gf_minus_gl':(d.gf-d.gl).mean(),'ratio':(d.gf/d.gl).mean(),'sd':(d.gf-d.gl).std()})).to_string())
# finger schedule when off insulin: gap to next finger reading given last reading and on/off insulin
fing = []
A = Gtr['gf'].values; I = Gtr['ins'].values
for i in range(5000):
    hh = np.where(~np.isnan(A[i]))[0]
    for a,b in zip(hh[:-1],hh[1:]): fing.append((A[i,a], b-a, not np.isnan(I[i,a])))
F = pd.DataFrame(fing, columns=['g','gap','on'])
print(F.groupby([pd.cut(F.g,[0,100,130,150,180,500]),'on'],observed=True).gap.agg(['size','mean',lambda x: x.value_counts().head(3).to_dict()]).to_string())
```