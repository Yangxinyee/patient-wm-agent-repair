The architecture fixes where an edit can act:

- **Value head:** a small MLP (`attr_heads.value_bin`) plus `embed_transition.value_bin`. The latter is a per-code transition lookup of logit offsets keyed by the last value bin.
- **Dose input:** dose enters only through the action atoms' `embed.numeric` columns (z_dose, z_rate), the `embed.bins.dose`/`embed.bins.rate` tables (shared across codes), and the action encoder, then flows through the recurrent core into the state the heads read.

A key fact from run 41: the hour-13 shift (−0.74 for insulin x2) is a pure one-step teacher-forced response to the 12:05 dose. So the model's one-step sensitivity is about −0.42 mg/dL per u/h, against a kernel lag-1 value of −2.0.

There are two possible explanations, and they call for different edits:
- **(a) The model underfits the dose effect everywhere.** Then in factual data, the residual of the next glucose reading should fall with the dose's deviation from the policy's expected dose for that reading, at a slope of about −1.6 per u/h. Adding the missing sensitivity, centred on the policy's expected dose, would then also improve factual NLL.
- **(b) The model fits dose variation within the policy's range but goes flat beyond it.** Then the residual slope is about 0, and the edit should mainly fix extrapolation.

Next, I will test this on the factual output I already have (fwd_2). I'll take readings in the hour after an insulin infusion. I'll compute the dose deviation from E[dose | decision-reading band], estimated from the 30k training stays. Then I'll regress the model's residuals on it. A slope near −1.6 supports (a); a slope near 0 supports (b).

```python
fa2 = pd.read_csv('/work/fwd_2/factual_original.csv')
print(fa2.columns.tolist(), fa2.shape); print(fa2.head(5).to_string())
print(json.dumps(json.load(open('/work/fwd_2/summary.json')))[:1200])
```