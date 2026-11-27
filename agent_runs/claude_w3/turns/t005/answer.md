The panel is built. Each stay has at most one reading of each type and one record of each drug per hour. Every stay ends at hour 25.17. (The column labelled "missing fraction" is actually the fraction present: glucose_finger about 52–59% of hours, rising slowly with hour; glucose_lab 20%; map_arterial 46%; map_cuff 61%.)

Next is the dosing policy. For each hour I will carry forward the latest glucose_finger and glucose_lab (and MAP for norepinephrine). Then I will tabulate the probability of a dose and the mean rate against those values and the previous hour's rate. If dosing is a sharp threshold rule on the latest reading, a threshold or discontinuity design may be usable. If it is smooth with lots of residual noise, I can adjust for history, provided dosing does not depend on hidden state.

```python
pn = panel.reset_index()
pn = pn[pn.h<=24].copy()
g = pn.groupby('stay_id')
for c in meas_codes:
    pn[c+'_last'] = g[c].ffill()
    # age of last
    t = pn['h'].where(pn[c].notna())
    pn[c+'_age'] = pn['h'] - t.groupby(pn.stay_id).ffill()
pn['ins_prev'] = g['insulin'].shift(1).fillna(0)
pn['ne_prev'] = g['norepinephrine'].shift(1).fillna(0)
pn['ins_on'] = (pn.insulin>0).astype(int); pn['ne_on']=(pn.norepinephrine>0).astype(int)
# policy vs last finger glucose
bins = [0,100,120,140,150,160,170,180,190,200,220,250,400]
pn['gfb'] = pd.cut(pn.glucose_finger_last, bins)
tab = pn[pn.h>=0].groupby('gfb', observed=True).agg(n=('ins_on','size'), p_on=('ins_on','mean'),
          rate_on=('insulin', lambda x: x[x>0].mean()), p_on_prev=('ins_prev', lambda x:(x>0).mean()))
print(tab.round(3).to_string())
mb = [0,40,50,55,60,65,70,75,80,90,200]
pn['mab'] = pd.cut(pn.map_arterial_last.fillna(pn.map_cuff_last), mb)
tab2 = pn[pn.h>=0].groupby('mab', observed=True).agg(n=('ne_on','size'), p_on=('ne_on','mean'),
          rate_on=('norepinephrine', lambda x: x[x>0].mean()))
print(tab2.round(3).to_string())
# stay-level: fraction of stays on insulin by whether ever on
st = pn.groupby('stay_id').agg(any_ins=('ins_on','max'), n_ins=('ins_on','sum'), any_ne=('ne_on','max'), n_ne=('ne_on','sum'),
                                gmean=('glucose_finger','mean'), mmean=('map_arterial','mean'), cmean=('map_cuff','mean'),
                                n_gf=('glucose_finger', lambda x: x.notna().sum()), n_ma=('map_arterial', lambda x: x.notna().sum()))
print(st.groupby('any_ins')[['n_ins','gmean','n_gf']].describe().round(1).T.to_string())
print(st.groupby('any_ne')[['n_ne','mmean','cmean','n_ma']].describe().round(1).T.to_string())
```