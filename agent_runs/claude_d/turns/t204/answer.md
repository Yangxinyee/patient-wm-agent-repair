The E2 rollout moved every readout in the right direction, but unevenly:

| edit | original | E2 | target |
|---|---|---|---|
| insulin ×2 | -1.73 | -7.17 | -14.0 |
| insulin ×½ | +0.91 | +1.47 | +7.0 |
| NE ×2 | -0.08 | +0.46 | +4.85 |
| NE ×½ | -0.48 | -1.92 | -2.43 |

So the gated channels work in the rollout and feedback does build up. But the responses are strongly asymmetric, whereas the true effects are linear (×2 : ×½ = -2 : 1).

My guess for the cause:

- **NE ×2** pushes rates past the top of the factual range (the policy caps at 1.0), where the state's dose representation saturates.
- **Insulin ×½** moves rates into the coarse low-rate bins, where the representation also seems compressed.

To see where the response falls short, I will open the per-stay-hour CSV for the edited model. I will regress the hour-13 shift, which has no feedback yet, on the hour-12 dose change for each edit. That gives the one-step gain per unit dose for doubling versus halving and shows whether the asymmetry is already present at the first step.

```python
d8 = pd.read_csv('/work/fwd_8/dose_edit_edited.csv'); o8 = pd.read_csv('/work/fwd_8/dose_edit_original.csv')
print(d8.shape, list(d8.columns)[:20])
print(d8.head(3).to_string()[:600])
```