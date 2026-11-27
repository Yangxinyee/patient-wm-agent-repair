Forward pass 9 (factual, original vs writer v2, 99 train stays) shows the edit is nearly inert: value NLL changed by +0.00009 and every event's MAE is unchanged. The cost is negligible, but so is the effect, while the original model's dose response is about 6 to 13 times smaller than my data-based target. Before choosing a stronger gain, I'll measure how far the edit actually moves the predicted means: the spread of (edited − original) predicted mean per event, and its slope on the lag-1 dose innovation. If the slope is around −0.1 per unit for insulin against the ≈ −2 needed, the gains must rise by roughly 20 times, or the write direction needs changing.

```python
for k in ['innA_s','innB_s','innA_in','innB_in','predz','zo_R','ze_R','preE','r2_','zo_D','ze_D','dzD']:
    if k in globals():
        v=globals()[k]; print(k, getattr(v,'shape',type(v).__name__))
dm=fe.predicted_mean.values-fo.predicted_mean.values
t=pd.DataFrame({'ev':fo.event.values,'dm':dm})
print(t.groupby('ev').dm.agg(['mean','std',lambda x: np.abs(x).mean()]).round(4).to_string())
```