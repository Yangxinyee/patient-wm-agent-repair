The design is fully regular. Every stay runs from hour −2 (hospital admission) to 25.17. Readings fall on whole hours and infusions start 5 minutes after the hour. Insulin rates sit on a 0.5 u/h grid up to 14.5. Norepinephrine rates run 0.01–1.00, with a hard cap at 1.00 (33k records). About 61% of stays get insulin and 44% get norepinephrine.

The cap matters for the evaluation. Norepinephrine x2 pushes many rates above anything in training (bin 16 and z up to about 2.6), so the model must extrapolate there.

Next I'll pivot each stay onto an hourly grid (glucose readings, MAP readings, drug rates) and save it. That lets me study how doses are chosen and what follows them. I'll also start the notebook.

```python
def grid(df):
    df = df.copy(); df['h'] = np.floor(df.hour+1e-6).astype(int)
    g = {}
    for ev, col, key in [('glucose_finger','value','gf'),('glucose_lab','value','gl'),('map_arterial','value','ma'),
                         ('map_cuff','value','mc'),('insulin','rate','ins'),('norepinephrine','rate','ne')]:
        s = df[df.event==ev]
        g[key] = s.pivot_table(index='stay_id', columns='h', values=col, aggfunc='first')
    stays = sorted(df.stay_id.unique()); hours = list(range(-1,26))
    G = {k: v.reindex(index=stays, columns=hours) for k,v in g.items()}
    return stays, G
stays_tr, Gtr = grid(tr); stays_tu, Gtu = grid(tu)
for k,v in Gtr.items(): print(k, 'non-null frac by hour:', np.round(v.notna().mean().values,2))
import pickle; pickle.dump((stays_tr,Gtr,stays_tu,Gtu), open('/work/grid.pkl','wb'))
open('/work/notes.md','w').write("""# Notebook
## Facts (runs 1-5)
- Stays: hosp adm -2h, ICU adm 0h, end 25.1667h. Readings at integer hours; insulin/NE records at h+0.0833, run 1h.
- Insulin rate grid 0.5 u/h (0.5..14.5), dose=rate. NE rate 0.01..1.00 (cap 1.00, 33k recs), dose=4.8*rate.
- 18335/30000 stays have insulin, 13174 have NE. NE x2 pushes rates >1 (never seen in training, bin 16).
- Grid saved /work/grid.pkl: (stays_tr,Gtr,stays_tu,Gtu), Gtr[key] DataFrame stay x hour(-1..25), keys gf gl ma mc ins ne
""")
```