At hour 13, which is still effectively teacher-forced, the edited model's glucose prediction sits above the original's by +2.37 on average (sd 2.06, mostly positive). My design should centre this offset near zero, so the edit adds a systematic upward glucose bias. Next I'll regress this offset on the recorded 12:05 insulin rate, the 12:00 glucose and the NE rate, and do the same for MAP.
- If the offset is present with zero insulin, the gate leaks or the quiet directions are not quiet on these stays.
- If it scales with dose with the wrong sign, the decoder or carrier is reversed on the tool.

```python
def at_hour(df, ev, h0, h1, col='value'):
    s = df[(df.event==ev)&(df.hour>=h0)&(df.hour<h1)]
    return s.groupby('stay_id')[col].first()
ins12 = at_hour(tun,'insulin',12,13,'rate'); ne12 = at_hour(tun,'norepinephrine',12,13,'rate')
gf12 = at_hour(tun,'glucose_finger',12,12.5); gl12 = at_hour(tun,'glucose_lab',12,12.5)
ma12 = at_hour(tun,'map_arterial',12,12.5); mc12 = at_hour(tun,'map_cuff',12,12.5)
T13 = g13.set_index('stay_id')
T13['ins'] = ins12.reindex(T13.index).fillna(0); T13['ne'] = ne12.reindex(T13.index).fillna(0)
T13['g'] = gf12.reindex(T13.index).fillna(gl12.reindex(T13.index))
print(T13[['d','ins','ne','g']].corr().round(3))
print(T13.groupby(T13.ins>0).d.describe())
# MAP rows
m13 = de_o[(de_o.hour==13)&(de_o.edit=='ne_x2')][['stay_id','factual']].merge(
      de_e[(de_e.hour==13)&(de_e.edit=='ne_x2')][['stay_id','factual']], on='stay_id', suffixes=('_o','_e')).set_index('stay_id')
m13['d'] = m13.factual_e - m13.factual_o
m13['ne'] = ne12.reindex(m13.index).fillna(0); m13['ins'] = ins12.reindex(m13.index).fillna(0); m13['map']=ma12.reindex(m13.index).fillna(mc12.reindex(m13.index))
print(m13[['d','ne','ins','map']].corr().round(3)); print(m13.groupby(m13.ne>0).d.describe())
print(tun.event.unique())
```