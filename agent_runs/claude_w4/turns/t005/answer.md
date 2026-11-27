## Step 4: the dosing policy

**What the last result showed.**
- Every stay is exactly 25.17 hours long, with ICU hours 0 to 25 and hospital admission at hour -2.
- Readings are on the hour; drug records start 5 minutes later.
- Insulin rates are on a 0.5 grid (0.5 to 11.5, mode 1.0).
- Norepinephrine rates are on a 0.01 grid, with a hard cap at 1.0: 14% of norepinephrine records sit exactly at the cap.
- About 61% of stays have insulin and 44% have norepinephrine. Arterial MAP is recorded in only about half of stays, perhaps the sicker ones; cuff MAP and finger glucose are recorded in almost all.

**Why the cap matters.** Rate = clip(f(state), 0.01, 1.0) means the dose is insensitive to the patient's condition at the cap. That could become an identification tool later.

**What I will do next.** Look at a few stays in full, then tabulate how each rate relates to:
- the reading just before it (finger glucose or MAP at the same integer hour);
- the previous rate.

**What would count as evidence.**
- Clean thresholds would point to a protocol.
- A strong dependence on the previous rate plus the reading (titration) would show how much unexplained "noise" is left.
- A large unexplained part could be either random (useful) or driven by hidden state (confounding).

```python
pd.set_option('display.width', 250)
# examine one NE stay and one insulin stay in wide format
def wide(df, sid):
    x = df[df.stay_id==sid].copy()
    x['h'] = np.floor(x.hour+1e-9).astype(int)
    w = x.pivot_table(index='h', columns='event', values='value', aggfunc='first')
    r = x[x.event.isin(['insulin','norepinephrine'])].pivot_table(index='h', columns='event', values='rate', aggfunc='first')
    r.columns = [c+'_rate' for c in r.columns]
    return w.join(r, how='outer')
ne_sids = ne.stay_id.unique()
print(wide(tr, ne_sids[0]).to_string())
print(wide(tr, ne_sids[1]).to_string())
```