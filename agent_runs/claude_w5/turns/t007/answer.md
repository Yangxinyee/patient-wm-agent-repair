This stay shows how regular the simulated ICU is:
- **Readings:** arterial line, finger-stick glucose and cuff readings on the hour, plus occasional glucose_lab.
- **Insulin:** rates in steps of 0.5 u/h, varying a lot from hour to hour (1.0 to 5.0) while glucose is around 180–250, so there seems to be considerable dosing noise.
- **Norepinephrine:** dose = 4.8 × rate exactly (an 80 kg patient, so dose carries no extra information), and the rate is capped at 1.0.
- **Monitoring follows treatment:** insulin stays get hourly finger-sticks and norepinephrine stays get arterial lines.

To estimate true effects I'll build an hourly panel per stay: readings at hour h, and insulin and norepinephrine rates at h + 5 min. Then I'll study the dosing policy, regressing the rate on the readings just observed and the previous rate.
- If the rate is a function of recorded readings plus independent noise, then regressing future glucose on dose with the history controlled identifies the causal effect.
- If the residual dose noise is correlated with future glucose beyond what the history explains, that signals hidden confounding, which I'll test later.

```python
import numpy as np, pandas as pd
def build_panel(df):
    df = df.copy()
    df['h'] = np.floor(df.hour+1e-6).astype(int)
    meas = df[df.event.isin(['glucose_finger','glucose_lab','map_arterial','map_cuff'])]
    P = meas.pivot_table(index=['stay_id','h'], columns='event', values='value', aggfunc='first')
    drugs = df[df.event.isin(['insulin','norepinephrine'])]
    D = drugs.pivot_table(index=['stay_id','h'], columns='event', values='rate', aggfunc='first')
    stays = df.stay_id.unique()
    idx = pd.MultiIndex.from_product([stays, range(-2,26)], names=['stay_id','h'])
    Pn = pd.DataFrame(index=idx).join(P).join(D)
    for c in ['glucose_finger','glucose_lab','map_arterial','map_cuff','insulin','norepinephrine']:
        if c not in Pn: Pn[c]=np.nan
    return Pn.reset_index()
P = build_panel(tr)
PT = build_panel(tu)
print(P.groupby('h')[['glucose_finger','glucose_lab','map_arterial','map_cuff','insulin','norepinephrine']].count().to_string())
```