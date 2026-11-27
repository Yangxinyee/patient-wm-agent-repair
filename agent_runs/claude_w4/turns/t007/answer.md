## Step 6: fix the panel code

**What the last result showed.** The panel built correctly (840,000 stay-hours). My summary code failed only because the column name `ne` clashes with pandas' built-in `.ne()` method.

**What I will do next.**
- Rename the drug columns to `ins_rate` and `ne_rate` and rerun the summaries: the hour ranges, how often an infusion switches on or off, when it starts, and how often each reading is taken.
- Also show the overall frequency of drug-on hours.

**What would count as evidence.** Few switches (0–1) would mean drugs, once started, continue to the end of the stay. Many switches would mean intermittent use.

```python
P = P.rename(columns={'ins':'ins_rate','ne':'ne_rate'}); PT = PT.rename(columns={'ins':'ins_rate','ne':'ne_rate'})
g = P.reset_index(); gT = PT.reset_index()
print("drug hour range ins:", g[g.ins_rate>0].h.agg(['min','max']).to_dict(), " ne:", g[g.ne_rate>0].h.agg(['min','max']).to_dict())
for d in ['ins_rate','ne_rate']:
    on = (g[d].values>0).reshape(-1,28)
    sw = (np.diff(on.astype(int),axis=1)!=0).sum(1)
    print(d, "switch count dist:", pd.Series(sw).value_counts().sort_index().head(12).to_dict())
    anyon = on.any(1)
    st = np.argmax(on,axis=1)[anyon]-2
    print(d, "start hour dist:", pd.Series(st).value_counts().sort_index().to_dict())
    print(d, "frac on per hour (h=-2..25):", np.round(on.mean(0),3).tolist())
print(g[g.h.between(0,24)][['glucose_finger','glucose_lab','map_arterial','map_cuff']].notna().mean())
```