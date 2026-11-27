This changes the picture.

- **Starts are deterministic.** From hour 0 on, norepinephrine starts whenever MAP is below 65 and it was off. My earlier "55%" came from hour −1 rows, when no drug can be given.
- **Start dose and MAP.** Among starts at the same MAP reading, a higher starting dose goes with a much larger later MAP *fall*: −9.8 ± 0.4 mmHg per mcg/kg/min at 1 h, −9.4 at 2 h and −8.4 at 3 h. That is hugely significant, and opposite to what the titration surprise shows (about 0).
- **Stops and no-reading hours.** The stop and no-reading-hour checks show nothing notable.

Either norepinephrine has an effect only at initiation, which seems unlikely, or the starting dose is chosen using information that predicts where MAP is heading. That could be the recent trend or something hidden, which would make the start dose a confounded signal. A world model would exploit that signal, and a dose edit would then move its predictions in the wrong direction.

Next I will work out what sets the starting dose. I will regress the start dose on the current reading, the previous one or two readings (trend), the hour, the cuff-versus-arterial reading and the next reading. If the previous MAP (the trend) explains the start dose and the −9.8 vanishes once I control for it, the start dose is trend-driven confounding. If the start dose tracks only future MAP, the clinician uses hidden information.

```python
St['M_p1'] = P.map_arterial.groupby(level=0).shift(1).reindex(St.index).values
St['C_p1'] = P.map_cuff.groupby(level=0).shift(1).reindex(St.index).values
St['x_p1'] = np.where(St.xtype=='art', St.M_p1, St.C_p1)
St['hr'] = St.index.get_level_values(1)
St['lsd'] = np.log(St['ne'])
print(St['ne'].describe().round(3))
print(St.groupby(St.x.clip(40,64).round()//4*4)['ne'].agg(['size','mean','std']).round(3).T.to_string())
m = St.x_p1.notna()
S_ = St[m].copy(); S_['trend'] = S_.x - S_.x_p1
b,se,n = fe_slope(S_.lsd.values, S_.trend.values, S_.grp.values, S_.stay.values)
print(f"log start dose ~ trend (x - x_prev): {b:.4f}+-{se:.4f} n={n}")
b,se,n = fe_slope(S_.lsd.values, S_.x_p1.values, S_.grp.values, S_.stay.values)
print(f"log start dose ~ x_prev: {b:.4f}+-{se:.4f}")
for k in [1,2]:
    yk = S_[f'y{k}'].values
    b,se,n = fe_slope(yk, S_['ne'].values, (S_.grp + "_" + S_.x_p1.round().astype(int).astype(str)).values, S_.stay.values)
    print(f"k={k} per-unit NE controlling x_prev FE: {b:+.2f}+-{se:.2f} n={n}")
# does start dose correlate with the next reading itself (beyond current)? and hour?
b,se,n = fe_slope(St.lsd.values, St.hr.values.astype(float), St.grp.values, St.stay.values)
print(f"log start dose ~ hour: {b:.4f}+-{se:.4f}")
```