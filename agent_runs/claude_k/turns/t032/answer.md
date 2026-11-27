The CSV has one row per stay, hour and edit, giving the shift (mean over 20 paths), its SE, the factual predicted mean and a "reached" flag. "Reached" seems to mean the hour comes after the first infusion of that drug at or after hour 12. The listed hours seem to be those whose record has the readout reading (stay 2020000000 lists only hours 15–24).

Next I'll check that interpretation against the records: are the listed hours those with a glucose_finger (or map_arterial) reading, and does "reached" start right after the first infusion from hour 12? Then I'll compute my records-based true shift on exactly the reached rows, per edit, using the estimated impulse responses applied to the recorded doses. If the hours match and my truth is about 5–10× the model's shifts, I have a concrete target and know the gap.

```python
fw['reached'] = fw.reached.astype(bool)
sid_index = {s:i for i,s in enumerate(ids_t)}
gf_t = PT['glucose_finger'].reindex(ids_t).values; ma_t = PT['map_arterial'].reindex(ids_t).values
# check hours listed vs readings present
chk=[]
for (sid, ed), g in fw.groupby(['stay_id','edit']):
    i = sid_index[sid]; A = gf_t if ed.startswith('insulin') else ma_t
    present = [h for h in range(13,25) if np.isfinite(A[i, col(h)])]
    D = ins_t if ed.startswith('insulin') else ne_t
    first = next((h for h in range(12,25) if D[i,h]>0), None)
    reach_pred = [h for h in present if first is not None and h > first]
    chk.append((sorted(g.hour.tolist())==present, sorted(g[g.reached].hour.tolist())==reach_pred))
chk=np.array(chk); print("hours==present readings:", chk[:,0].mean(), " reached==after first dose:", chk[:,1].mean())
# my truth on reached rows
mults = {'insulin_x2':2.0,'insulin_half':0.5,'ne_x2':2.0,'ne_half':0.5}
S = {ed: true_shift(ins_t if ed.startswith('insulin') else ne_t, h_ins if ed.startswith('insulin') else h_ne, m) for ed,m in mults.items()}
fw['mytrue'] = [S[ed][sid_index[s], h] for s,h,ed in zip(fw.stay_id, fw.hour, fw.edit)]
summ = fw[fw.reached].groupby('edit').agg(model=('shift','mean'), mytrue=('mytrue','mean'), n=('shift','size'))
print(summ.round(3))
```