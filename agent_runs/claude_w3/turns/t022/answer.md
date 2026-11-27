The CSV shows exactly how the readout works:
- There is one row per stay, per hour from 13 to 24 with a recorded glucose_finger (or map_arterial) reading, per edit.
- "reached" marks hours after the first dose at or after hour 12. For example, a dose at 17.08 makes hour 17 unreached and hour 18 onward reached.
- The readout is the mean shift over reached rows.

So the true quantity is the effect at the recorded reading times in drug-exposed hours, and I should compute my estimate on exactly those rows.

Next I will apply the provisional insulin impulse response to the reached insulin rows of this file, using the recorded tuning doses. This gives a like-for-like provisional truth to compare with the model's −0.39. I will also look at the norepinephrine rows (which stays and hours they cover), so I know the target population for the norepinephrine effect.

```python
Sx = tup.stay_id.nunique()
ids_tu = tup.stay_id.values.reshape(Sx,25)[:,0]; pos = {s:i for i,s in enumerate(ids_tu)}
INS_tu = tup.insulin.values.reshape(Sx,25); NE_tu = tup.norepinephrine.values.reshape(Sx,25)
def truth_rows(de_edit, D, th, mult):
    vals = []
    for _, r in de_edit.iterrows():
        i = pos[r.stay_id]; t = int(r.hour)
        e = sum(th.get(t-s,0.0)*(mult-1)*D[i,s] for s in range(12,t))
        vals.append(e)
    return np.array(vals)
for ed, mult in [('insulin_x2',2.0),('insulin_half',0.5)]:
    rr = de[(de.edit==ed)&(de.reached)]
    tv = truth_rows(rr, INS_tu, theta, mult)
    print(ed, " model:", rr['shift'].mean().round(3), " prelim truth:", tv.mean().round(2), " rows", len(rr))
ner = de[(de.edit=='ne_x2')]
print("ne rows:", len(ner), " reached:", ner.reached.sum(), " stays:", ner.stay_id.nunique())
# do ne rows correspond to arterial readings?
tu_art = tu[tu.event=='map_arterial'].copy(); tu_art['h']=tu_art.hour.round().astype(int)
art_set = set(zip(tu_art.stay_id, tu_art.h))
print("frac ne rows with recorded arterial reading at that hour:", np.mean([(s,h) in art_set for s,h in zip(ner.stay_id, ner.hour)]))
tu_gf = tu[tu.event=='glucose_finger'].copy(); tu_gf['h']=tu_gf.hour.round().astype(int)
gf_set = set(zip(tu_gf.stay_id, tu_gf.h))
insr = de[de.edit=='insulin_x2']
print("frac ins rows with recorded finger reading:", np.mean([(s,h) in gf_set for s,h in zip(insr.stay_id, insr.hour)]))
```