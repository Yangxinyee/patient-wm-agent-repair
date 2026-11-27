Every reading falls on a whole hour. Every infusion record starts 5 minutes after the hour (h + 0.0833) and runs one hour. So each hour the clinician sees the readings at hour h and sets the insulin and norepinephrine rates for the coming hour. The next reading, 55 minutes later, is the first that can show an effect.

Two other facts from the first stays: finger glucose seems to read about 30 mg/dL above lab glucose, and cuff MAP about 8 mmHg above arterial. Stays run from hour −2 to 24. About 62% of stays get insulin and 44% get norepinephrine, mostly from the first hour.

Next I will start the notebook and build an hour-by-stay panel of readings and rates. That panel is what I need to model the dosing policy: how much of each rate is explained by the visible readings, and how much variation is left over that could identify the drug effects.

```python
notes = """# NOTES (world-model dose repair)

## Setup facts (runs 1-5)
- Data: hourly grid. Readings at integer hours -1..24; infusion records start at h+0.0833 (5 min after the hour), run 1h.
  Stay: hospital_adm at -2, icu_adm at 0, ends 25.1667. 30k train stays, 400 tuning.
- 62% stays get insulin, 44% NE, 27% both. Insulin rate 0.5..18.5 step 0.5 (median 2); NE rate 0.01..1.0 (capped at 1.0),
  NE dose = rate*80kg*60/1000 (dose bin edges top at 4.8 -> rate 1.0).
- Shared tables: embed.bins.dose/rate (17x192), embed.numeric (192x6: z_value,has_value,z_dose,has_dose,z_rate,has_rate)
  are shared by insulin (code 8) and NE (code 9). Code-specific edits must go via encode_action nonlinearity or downstream.
- Doubling NE rate -> above 1.0 = out of training range (bin 16, z>0.77).
"""
open('/work/notes.md','w').write(notes)

# Build hourly panel
df = tr.copy()
df['h'] = np.floor(df.hour+1e-6).astype(int)
def panel(df):
    meas = df[df.event.isin(['glucose_finger','glucose_lab','map_arterial','map_cuff'])]
    P = meas.pivot_table(index=['stay_id','h'], columns='event', values='value', aggfunc='first')
    drugs = df[df.event.isin(['insulin','norepinephrine'])]
    D = drugs.pivot_table(index=['stay_id','h'], columns='event', values='rate', aggfunc='first')
    stays = df.stay_id.unique()
    idx = pd.MultiIndex.from_product([stays, range(-2,25)], names=['stay_id','h'])
    out = pd.DataFrame(index=idx).join(P).join(D)
    return out
PN = panel(df)
print(PN.shape)
print(PN.notna().mean())
print(PN.groupby(level='h').apply(lambda x: x.notna().mean()).round(2).to_string())
```