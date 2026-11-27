## Step 7: what drives insulin and norepinephrine dosing

**What the last result showed.**
- About 31% of stays are on each drug from hour 0. The rest start at various hours.
- Insulin often stops and restarts (2–4+ on/off switches). Norepinephrine is usually a single run.
- Finger glucose is measured in about 56% of hours, lab glucose in 20%, arterial MAP in 46%, cuff MAP in 61%.

**What I will do next.** Fit the dosing policy. For each stay-hour, carry forward:
- the most recent glucose readings (finger and lab) and the age of each;
- the most recent MAP readings;
- the previous rates.

Then look at:
- insulin on/off and rate against the last glucose, by bins;
- norepinephrine against the last MAP;
- how much of the rate variation these explain (R²).

**What would count as evidence.**
- A low R² with an independent, noise-like residual would suggest dose variation that is as good as random.
- Sharp jumps at specific glucose or MAP values would indicate protocol thresholds.

```python
def add_last(P):
    g = P.reset_index().copy()
    for c in ['glucose_finger','glucose_lab','map_arterial','map_cuff']:
        g['last_'+c] = g.groupby('stay_id')[c].ffill()
        hh = g.h.where(g[c].notna())
        g['age_'+c] = g.h - hh.groupby(g.stay_id).ffill()
    for d in ['ins_rate','ne_rate']:
        g['prev_'+d] = g.groupby('stay_id')[d].shift(1).fillna(0)
    return g
g = add_last(P); gT = add_last(PT)
x = g[g.h.between(0,24)]
x['gbin'] = pd.cut(x.last_glucose_finger, [0,100,120,140,150,160,170,180,190,200,210,220,240,260,400])
print(x.groupby('gbin').agg(n=('ins_rate','size'), on=('ins_rate', lambda s:(s>0).mean()), mean_rate_on=('ins_rate', lambda s: s[s>0].mean())).to_string())
x['mbin'] = pd.cut(x.last_map_cuff, [0,40,50,55,60,62,64,65,66,68,70,75,80,90,200])
print(x.groupby('mbin').agg(n=('ne_rate','size'), on=('ne_rate', lambda s:(s>0).mean()), mean_rate_on=('ne_rate', lambda s: s[s>0].mean())).to_string())
x['abin'] = pd.cut(x.last_map_arterial, [0,40,50,55,60,62,64,65,66,68,70,75,80,90,200])
print(x.groupby('abin').agg(n=('ne_rate','size'), on=('ne_rate', lambda s:(s>0).mean()), mean_rate_on=('ne_rate', lambda s: s[s>0].mean())).to_string())
```